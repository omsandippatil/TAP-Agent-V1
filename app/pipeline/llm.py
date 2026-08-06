import asyncio
import json
import logging
import re
import time
import typing

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.config import settings
from app.pipeline.textproc import combine_evidence_text, estimate_tokens

logger = logging.getLogger("tap.llm")

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"

LLM_UNAVAILABLE_EVIDENCE = "LLM unavailable — unable to generate evidence"

OUTPUT_TOKEN_RESERVE = 6000
EXTRACTION_OUTPUT_TOKEN_RESERVE = 3200
SCAFFOLD_SAFETY_MARGIN = 400
MIN_EVIDENCE_TOKEN_BUDGET = 500
ANTHROPIC_REQUEST_TIMEOUT_SECONDS = 120.0
INTER_CALL_DELAY_SECONDS = 1.5
MIN_PROMPT_TRIM_CHARS = 150
MAX_PROMPT_SHRINK_ATTEMPTS = 6
PROMPT_SHRINK_SAFETY_MARGIN = 120

DEFAULT_MISSION = (
    "The Apprentice Project (TAP) develops 21st-century skills (critical thinking, "
    "creativity, confidence, communication, problem-solving, self-awareness, "
    "financial literacy) for low-income middle and high school students in India, "
    "delivered through TAP Buddy — an AI-powered WhatsApp chatbot with video "
    "electives (Coding, Science, Visual Arts, Financial Literacy). TAP works "
    "exclusively in government schools with partners like MCD, DoE Delhi, BMC "
    "Mumbai and SCERT Maharashtra. TAP does NOT run vocational training or job "
    "placement."
)

CRITERIA_IDS = [
    "education_intervention",
    "stem",
    "tech_21cs",
    "public_schooling",
    "systems_change",
    "programme_depth",
    "partnership_quality",
    "decision_maker_accessibility",
    "csr_trajectory",
    "delivery_model_fit",
    "outreach_readiness",
    "funding_capacity",
    "csr_spend_trend",
    "decision_maker_tenure",
    "group_foundation_routing",
    "board_education_affinity",
    "employee_volunteering",
]

CRITERIA_TITLES = {
    "education_intervention": "Education: intervention not scholarship",
    "stem": "STEM exposure",
    "tech_21cs": "Technology & 21st-century skills",
    "public_schooling": "Public-schooling understanding",
    "systems_change": "Systems-change orientation",
    "programme_depth": "Programme maturity & depth",
    "partnership_quality": "NGO partnership quality",
    "decision_maker_accessibility": "Decision-maker accessibility",
    "csr_trajectory": "CSR trajectory (growing / flat / shrinking)",
    "delivery_model_fit": "Delivery-model fit for TAP entry",
    "outreach_readiness": "Outreach readiness (open call / RFP / warm channel)",
    "funding_capacity": "Funding capacity vs TAP's typical ask size",
    "csr_spend_trend": "Multi-year CSR spend trend",
    "decision_maker_tenure": "CSR-head tenure (newly appointed vs entrenched)",
    "group_foundation_routing": "CSR routed through a group/parent foundation",
    "board_education_affinity": "Board or promoter personal education-philanthropy ties",
    "employee_volunteering": "Employee volunteering / payroll-giving programmes",
}

CRITERIA_WEIGHTS = {
    "education_intervention": 12, "stem": 8, "tech_21cs": 10, "public_schooling": 10,
    "systems_change": 8, "programme_depth": 8, "partnership_quality": 6,
    "decision_maker_accessibility": 4, "csr_trajectory": 4, "delivery_model_fit": 8,
    "outreach_readiness": 4, "funding_capacity": 4, "csr_spend_trend": 4,
    "decision_maker_tenure": 3, "group_foundation_routing": 3,
    "board_education_affinity": 2, "employee_volunteering": 2,
}
assert set(CRITERIA_WEIGHTS) == set(CRITERIA_IDS)
assert sum(CRITERIA_WEIGHTS.values()) == 100

_RUBRIC = {
    "education_intervention": "hands-on programme, not a scholarship or one-off donation",
    "stem": "named STEM/coding/robotics/science exposure",
    "tech_21cs": "tech-delivered learning or 21st-century-skills content",
    "public_schooling": "explicit government-school work; absence alone doesn't disqualify",
    "systems_change": "teacher training, measured outcomes, scale, or policy influence",
    "programme_depth": "one-off activity scores lower; named multi-year programme scores higher",
    "partnership_quality": "named, multi-year NGO partner scores higher; give real credit if the company already funds other education/skilling-adjacent NGOs, even ones unrelated to TAP",
    "decision_maker_accessibility": "a named individual with a current CSR-decision title",
    "csr_trajectory": "expansion scores higher, flat scores medium, contraction scores lower, no signal takes the sector default",
    "delivery_model_fit": "how cleanly TAP could enter as a grantee or as a delivery partner",
    "outreach_readiness": "an open call or RFP scores high; a closed/invite-only programme scores low",
    "funding_capacity": "whether the disclosed or plausibly-estimated CSR budget could cover a TAP-sized grant",
    "csr_spend_trend": "rising multi-year spend scores high, flat scores medium, declining scores low, no data takes the sector default",
    "decision_maker_tenure": "a recently appointed CSR head is a positive signal (new mandate); entrenched or unknown tenure is neutral",
    "group_foundation_routing": "a named parent/group foundation handling CSR scores high; no signal is a low but non-zero baseline",
    "board_education_affinity": "a named board/promoter personal history with education philanthropy scores high; generic or none is a low baseline, not zero",
    "employee_volunteering": "an actively named education-linked volunteering programme scores high; generic or none is a low baseline, not zero",
}


def _rubric_block() -> str:
    return "\n".join(f"- {key}: {value}" for key, value in _RUBRIC.items())


def _criteria_json_template() -> str:
    return ",\n".join(
        f'    {{"id": "{cid}", "name": "{CRITERIA_TITLES[cid]}", "score": <0-5>, "confidence": <0-100>, "evidence": "<short paraphrase>", "reasoning": "<short>"}}'
        for cid in CRITERIA_IDS
    )


# --------------------------------------------------------------------------
# MODE CALIBRATION
#
# Screen and deep research are different questions, not the same question run
# with less evidence. Screen mode exists to triage — its job is "does this
# company deserve a deep-research pass", so it should read genuinely promising
# but thinly-documented signals generously. Deep mode exists to brief someone
# on an actual outreach decision, so it stays strict and evidentiary. Treating
# them identically (the old behaviour) meant every screen-mode run was
# structurally penalised for having less evidence to work with, even when the
# available evidence was itself positive — this is what was producing
# uniformly low screen scores regardless of true fit.
# --------------------------------------------------------------------------

MODE_CALIBRATION = {
    "screen": {
        "stance": (
            "MODE: SCREEN (triage pass). Your job is to judge whether {company} is worth a "
            "full deep-research pass, not to produce a final outreach-ready verdict. Screen "
            "mode structurally sees fewer sources than deep mode — that is expected, not a "
            "flaw in the company. Read genuinely promising but thinly-documented signals "
            "generously: a company with clear education/CSR activity and no disqualifying "
            "red flag should score in a range that reads as 'worth a deep dive', even if "
            "spend figures, named partners, or a named decision-maker haven't surfaced yet. "
            "Reserve low scores for cases with an actual negative signal (no CSR at all, "
            "explicitly non-education CSR, or a stated policy against NGO partnerships) — "
            "thin sourcing alone is never sufficient reason for a low score in screen mode."
        ),
        "authenticity_cap_threshold": 15,
        "authenticity_cap_ceiling": 78,
    },
    "deep": {
        "stance": (
            "MODE: DEEP RESEARCH (outreach-ready brief). This analysis will inform an actual "
            "outreach decision, so hold evidence to a stricter standard than a triage pass — "
            "undocumented should still be read generously per the philosophy below, but "
            "claims should be well-grounded in what was actually fetched, since a person may "
            "act on this directly."
        ),
        "authenticity_cap_threshold": 30,
        "authenticity_cap_ceiling": 65,
    },
}


def _mode_calibration(mode: str) -> dict:
    return MODE_CALIBRATION.get(mode, MODE_CALIBRATION["deep"])


SCORING_PHILOSOPHY = (
    "SCORING PHILOSOPHY: most CSR activity in India is only partially documented online. "
    "Silence about something is not evidence against it. Never score a criterion at 0, and "
    "never treat a company as a poor fit, purely because a fact wasn't surfaced by the sources "
    "you were given — reserve 0 only for evidence that actively contradicts fit (e.g. an "
    "explicit statement the company does not run education CSR). Where the record is quiet on "
    "a criterion, score it from sector, scale, and adjacent CSR behavior visible elsewhere in "
    "the evidence, and label it in your `evidence` text as an inferred estimate, not a "
    "confirmed absence. When genuinely torn between two adjacent scores for a criterion, prefer "
    "the higher one — undocumented should never read as a negative signal. This applies "
    "per-criterion; it does not mean invent facts, and it does not mean treat every company as "
    "a great fit — evidence that actively points away from fit should bring scores down "
    "honestly, just as evidence that supports fit should bring them up."
)

SPEND_VS_REVENUE_RULE = (
    "SPEND VS REVENUE (common error, check this carefully): revenue, turnover, net worth, "
    "profit, market cap, and EBITDA describe business SCALE, never CSR spend. Never place them "
    "in spend.display or spend.inr_crore, and never call them CSR spend/budget/fund. Set "
    "spend.has_disclosed_budget=true only for a figure explicitly labeled as CSR "
    "expenditure/spend/budget, or a stated CSR-mandate percentage applied to a stated profit "
    "figure. Otherwise has_disclosed_budget=false and inr_crore=null. Put any business-scale "
    "figures only in eligibility.net_worth_turnover_signal (as text) and the "
    "eligibility.net_worth_turnover_inr_crore / eligibility.net_profit_inr_crore numeric fields "
    "— never in spend."
)

EDUCATION_SPEND_RULE = (
    "EDUCATION SPEND VS TOTAL CSR (second common error): most companies run CSR across several "
    "causes, not just education. spend.inr_crore / spend.display / spend.fiscal_year / "
    "spend.trend_* must hold ONLY the education-specific slice — set is_education_specific=true "
    "and populate these only when the evidence states an education-specific figure or "
    "percentage. If the evidence only gives a TOTAL CSR figure with no education breakdown, "
    "leave spend.is_education_specific=false and spend.inr_crore/display/fiscal_year empty, and "
    "put that total in total_csr_inr_crore / total_csr_display / total_csr_fiscal_year instead, "
    "clearly labeled as total CSR, never as the education budget."
)

PARTNER_RULE = (
    "PARTNERS: include a named third-party organisation as a partner only if the evidence shows "
    "an actual relationship (funds, co-designs, implements with, partners with, delivers via, "
    "works with). Use confidence='confirmed' if the relationship verb is explicit and clear, "
    "confidence='probable' if the org is named alongside the company in a CSR/education context "
    "but the relationship language is vague or implied. Do not include internal campaign names "
    "(they are not organisations), generic unnamed government references, or award/certifying "
    "bodies. For each partner, also fill programme/year/geography if the evidence states them "
    "(leave empty otherwise — never guess), and set similar_to_tap_profile=true if the partner "
    "org is itself an education/skilling/government-school-facing NGO or intermediary (even if "
    "unrelated to TAP) — this is a genuine positive signal for partnership_quality and "
    "delivery_model_fit, so weigh it there yourself. If your own fit_rationale or "
    "delivery_model_evidence text names a third-party org the company works with, that same org "
    "must also appear in the partners array — never describe a partnership in prose while "
    "leaving it out of the structured list."
)

PROGRAMME_RULE = (
    "PROGRAMMES: include a named programme only if you can state what is funded and who "
    "benefits, using whatever specificity the evidence actually gives — ordinary phrasing like "
    "'government-school students' or 'children' is fine, it doesn't need to be unusually "
    "precise. A bare theme mention with no named programme and nothing else to anchor it (e.g. "
    "just 'the company supports financial literacy', no name, no beneficiary, no funded "
    "activity) should be left out rather than invented into a full entry. confidence='confirmed' "
    "if there's one additional concrete supporting detail (scale, duration, since-when) beyond "
    "name/what's-funded/beneficiary; confidence='probable' otherwise. If your own fit_rationale "
    "or delivery_model_evidence names a specific initiative, it must also appear in the "
    "programmes array."
)

SOURCE_INTEGRITY_RULE = (
    "SOURCE INTEGRITY: before treating a fragment as evidence, confirm it's a genuine "
    "descriptive sentence about the company's own activity — not a nav menu, link list, or "
    "heading run-on. Confirm any excerpt is actually about the company being analysed and not a "
    "different entity that happens to share the page (this matters especially for "
    "people-search/LinkedIn snippets — if a profile's employer is a different company, that "
    "text is not evidence about this company even if this company's name appears elsewhere on "
    "the page). A person's individual career history describes that person, never a company "
    "programme."
)

EVIDENCE_STYLE_RULE = (
    "EVERY field must trace to the evidence you were given — never invent facts, and state "
    "partial evidence as partial. Evidence/reasoning fields are short paraphrases (under 20 "
    "words), never verbatim quotes except for exact figures, partner names, or programme names."
)

HIGHLIGHT_RULE = (
    "HIGHLIGHT: in fit_rationale, alignment_rationale, delivery_model_evidence, "
    "source_quality_assessment, csr_head_note, evidence_recency, contact_pathway.channel, "
    "strategic_insight, and each criterion's evidence — bold exactly one 2-3 word "
    "decision-relevant phrase with **asterisks**. Never bold a full sentence, a lone number, or "
    "more than 3 words. Never bold names, titles, sources, URLs, booleans, or enums. Skip only "
    "if the field is empty."
)


# --------------------------------------------------------------------------
# PASS 1 — EXTRACTION
#
# Pulls every structured fact (spend, programmes, partners, decision-makers,
# geographies, governance signals, sector/eligibility, red flags) out of the
# evidence with no scoring involved. Separating this from scoring means the
# scoring pass reads a short, already-distilled fact sheet instead of raw
# evidence text plus 20 instructions at once, which is what was causing
# scores to drift and undercount real signal in the single-shot version.
# --------------------------------------------------------------------------

def _extraction_prompt(company: str, mission: str, evidence_text: str, sources_manifest: str) -> str:
    return f"""You are a meticulous fact-extraction analyst. Extract every concrete, sourced fact about {company}'s India CSR activity from the evidence below. Do NOT score or judge fit — that happens in a separate pass. Your only job here is complete, accurate, well-cited extraction.

NGO MISSION (context only, for judging what counts as education-relevant — do not score against it here): {mission}

EVIDENCE (from sources actually fetched for {company} — numbered sources below can be cited):
\"\"\"
{evidence_text}
\"\"\"

SOURCES:
{sources_manifest}

{SPEND_VS_REVENUE_RULE}

{EDUCATION_SPEND_RULE}

{PARTNER_RULE}

{PROGRAMME_RULE}

{SOURCE_INTEGRITY_RULE}

{EVIDENCE_STYLE_RULE}

Extract, in this order:
1. delivery_model (FUNDER/IMPLEMENTER/HYBRID/UNCLEAR) + delivery_model_evidence (1 sentence).
2. spend — apply the SPEND VS REVENUE and EDUCATION SPEND rules strictly.
3. programmes[] — apply the PROGRAMME rule.
4. partners[] — apply the PARTNER rule, including similar_to_tap_profile.
5. decision_makers[] — every named leader/exec/spokesperson in a CSR/sustainability context; title, public_facing_score 0-100, tenure_status, linkedin_url only if a literal linkedin.com/in/ URL is present in the evidence.
6. geographies[] — every state/city explicitly named.
7. rfp_signal — an explicit call for NGO partners; default false/empty unless stated.
8. board_affinity — named board/promoter personal education-philanthropy history; default false/empty unless stated.
9. volunteering — named employee volunteering/payroll-giving touching education; default false/empty unless stated.
10. group_foundation — CSR run via a separate parent/group foundation, only if explicitly named.
11. eligibility — Section 135 applicability (LIKELY/UNLIKELY/UNKNOWN) from net worth/turnover/profit figures (kept separate from spend), plus the plain numeric business-scale fields.
12. sector — from company-description language; UNKNOWN only if truly no clue.
13. red_flags[] — genuine contradictions or marketing-not-substance signals, severity low/medium/high. Missing/undocumented details are NOT red flags.
14. contact_pathway — the single most concrete real channel; "Not identified" if nothing exists.
15. evidence_recency — one sentence on how current the evidence appears.
16. csr_head_note — one sentence, only from actual named-person context, never speculation from a bare title.
17. source_quality_assessment — 1-2 sentences: primary (company/regulator) vs secondary (press/snippets) sourcing.
18. overall_authenticity_score (0-100) — reflects sourcing quality (primary vs secondary, how many sources actually returned usable text), not evidence volume.
19. open_questions[] — up to 5 short, concrete, searchable items to verify.
20. key_facts_summary — 3-6 short bullet-style facts (as a single string, one per line prefixed with "- ") that most directly bear on education-CSR fit — this feeds directly into the scoring pass, so include anything that would move a fit judgment either up or down.

Reply with ONE JSON object, nothing else, no markdown fences.

JSON shape:
{{
  "delivery_model": "<FUNDER|IMPLEMENTER|HYBRID|UNCLEAR>",
  "delivery_model_evidence": "<sentence>",
  "spend": {{"inr_crore": <number or null, education-specific only>, "display": "<exact CSR-labeled education figure or empty>", "fiscal_year": "<if stated>", "is_education_specific": <bool>, "education_pct_of_total_csr": <number or null>, "has_disclosed_budget": <bool>, "confidence": <0-100>, "source_excerpt": "<short>", "trend_direction": "<RISING|FLAT|DECLINING|UNKNOWN>", "trend_evidence": "<short>", "history": [{{"fiscal_year": "<year>", "inr_crore": <number or null>, "display": "<as stated>", "source_excerpt": "<short>"}}], "total_csr_inr_crore": <number or null>, "total_csr_display": "<as stated or empty>", "total_csr_fiscal_year": "<if stated>"}},
  "programmes": [{{"name": "<exact name>", "what_is_funded": "<precise funded activity>", "beneficiary_group": "<named beneficiary group>", "beneficiary_type": "<SCHOOL_CHILDREN_CURRICULUM|ADULT|OTHER>", "description": "<short>", "is_multi_year": <bool>, "cohort_or_scale": "<if stated>", "source_excerpt": "<short>", "confidence": "<confirmed|probable>"}}],
  "partners": [{{"name": "<exact org name>", "relationship_type": "<funder|implementer|co-design|unclear>", "programme": "<or empty>", "year": "<or empty>", "geography": "<or empty>", "similar_to_tap_profile": <bool>, "source_excerpt": "<short, must show relationship language>", "confidence": "<confirmed|probable>"}}],
  "decision_makers": [{{"name": "<n>", "title": "<title>", "public_facing_score": <0-100>, "tenure_status": "<NEW_UNDER_1YR|ESTABLISHED_1_3YR|ENTRENCHED_3YR_PLUS|UNKNOWN>", "tenure_evidence": "<short>", "source_excerpt": "<short>", "linkedin_url": "<url or empty>"}}],
  "geographies": [{{"place": "<place>", "source_excerpt": "<short>"}}],
  "red_flags": [{{"flag": "<short label>", "severity": "<low|medium|high>", "explanation": "<short>"}}],
  "contact_pathway": {{"channel": "<sentence>", "evidence": "<short>"}},
  "rfp_signal": {{"present": <bool>, "channel": "<short>", "evidence": "<short>"}},
  "board_affinity": {{"present": <bool>, "person_name": "<name or empty>", "connection": "<short>", "source_excerpt": "<short>"}},
  "volunteering": {{"present": <bool>, "programme_name": "<name or empty>", "description": "<short>", "source_excerpt": "<short>"}},
  "group_foundation": {{"routed_through_group": <bool>, "foundation_name": "<name or empty>", "explanation": "<short>", "source_excerpt": "<short>"}},
  "eligibility": {{"plausibly_mandated": "<LIKELY|UNLIKELY|UNKNOWN>", "reasoning": "<short>", "net_worth_turnover_signal": "<short>", "net_worth_turnover_inr_crore": <number or null>, "net_profit_inr_crore": <number or null>}},
  "sector": {{"sector": "<sector>", "sub_sector": "<or empty>", "reasoning": "<short>"}},
  "evidence_recency": "<one sentence>",
  "csr_head_note": "<one sentence>",
  "source_quality_assessment": "<1-2 sentences>",
  "overall_authenticity_score": <int 0-100>,
  "open_questions": ["<short item>", "..."],
  "key_facts_summary": "<3-6 lines, each starting with '- '>"
}}"""


# --------------------------------------------------------------------------
# PASS 2 — SCORING
#
# Reads the distilled extraction output (not raw evidence) and produces the
# 17 criteria scores plus narrative fields. fit_score itself is NOT trusted
# from this call — it's computed deterministically in Python afterward from
# the weighted criteria (see _compute_fit_score), which is what actually
# fixes the "score doesn't match its own rubric" problem. The model is only
# asked for fit_score here as a sanity cross-check value for logging.
# --------------------------------------------------------------------------

def _scoring_prompt(company: str, mission: str, mode: str, extraction: dict, sources_manifest: str) -> str:
    calibration = _mode_calibration(mode)
    extraction_json = json.dumps(extraction, ensure_ascii=False, indent=2)
    return f"""You are a careful, fair-minded CSR partnerships analyst judging whether {company} is a good funding/partnership fit for an Indian education NGO. A separate extraction pass already pulled every fact below from the fetched evidence — do not re-extract or add new facts, only score against what's here and write the narrative fields.

{calibration['stance']}

NGO MISSION: {mission}

EXTRACTED FACTS FOR {company} (already verified against evidence — numbered sources below can be cited):
\"\"\"
{extraction_json}
\"\"\"

SOURCES:
{sources_manifest}

{SCORING_PHILOSOPHY}

{HIGHLIGHT_RULE}

Produce, in this order:
1. criteria[] — all 17 ids below, in order, each with id, name (copy exactly as given), score 0-5, confidence 0-100, short evidence, short reasoning, drawn only from the extracted facts above:
{_rubric_block()}
2. fit_score (int 0-100) — your own holistic cross-check estimate; the system will also compute a weighted score from your criteria and reconcile the two, so this does not need to be perfectly precise, but should be in the same ballpark as your criteria scores weighted by importance.
3. fit_rationale (2-4 sentences): justify the scoring from the extracted facts, stating plainly what's confirmed vs inferred vs undocumented. If a named partner/programme suggests a plausible but unconfirmed entry path, you may add one sentence starting literally "Inference (unconfirmed):" naming that specific org/programme — never invent one not in the extracted facts. If decision_makers and/or partners/programmes are non-empty, end with one short sentence "Key contacts: A (Title), B (Title); Key partners: X, Y" using only names from the extracted facts. Omit that closing sentence if both lists are empty.
4. overall_semantic_alignment (0-100) + alignment_rationale (1-2 sentences) — how well the company's actual activity matches the NGO mission semantically, independent of documentation completeness.
5. strategic_insight — a 150-280 word standalone narrative (this is the lead summary shown to the user first): measured and evidence-grounded, leading with genuine strengths before caveats, stating plainly whether/why this is a good fit, naming strongest/weakest dimensions without dwelling on the weakest, flagging group-foundation routing if present, noting eligibility if uncertain, and giving one concrete next step. If TAP-similar partners exist, mention that positively. {"Since this is a screen-mode pass, if the signal is promising but sourcing is thin, say plainly that a deep-research pass would surface more (spend figures, named partners, a decision-maker) rather than treating the gap as a weakness." if mode == "screen" else ""} End with the same "Key contacts: ...; Key partners: ..." sentence format as fit_rationale (only using names from the extracted facts), omitted if both lists are empty.

All criteria ids appear exactly once, in the order listed, each with its name copied exactly as given above. Keep every string concise so the full reply fits comfortably in your output budget.

Reply with ONE JSON object, nothing else, no markdown fences.

JSON shape:
{{
  "criteria": [
{_criteria_json_template()}
  ],
  "fit_score": <int 0-100>,
  "fit_rationale": "<2-4 sentences, one **2-3 word** highlight, optional Inference/Key-contacts clauses>",
  "overall_semantic_alignment": <int 0-100>,
  "alignment_rationale": "<1-2 sentences, one **2-3 word** highlight>",
  "strategic_insight": "<150-280 word narrative, one **2-3 word** highlight, optional Inference/Key-contacts clauses>"
}}"""


class CriterionResultSchema(BaseModel):
    id: str
    name: str = ""
    score: float = Field(ge=0, le=5)
    confidence: int = Field(ge=0, le=100)
    evidence: str = Field(default="", max_length=240)
    reasoning: str = Field(default="", max_length=240)
    source: str = Field(default="")


class SpendYearSchema(BaseModel):
    fiscal_year: str = ""
    inr_crore: float | None = None
    display: str = ""
    source: str = ""
    source_excerpt: str = Field(default="", max_length=200)


class SpendSchema(BaseModel):
    inr_crore: float | None = None
    display: str = ""
    fiscal_year: str = ""
    is_education_specific: bool = False
    education_pct_of_total_csr: float | None = None
    has_disclosed_budget: bool = False
    confidence: int = Field(ge=0, le=100, default=0)
    source_excerpt: str = Field(default="", max_length=200)
    source: str = ""
    trend_direction: str = "UNKNOWN"
    trend_evidence: str = Field(default="", max_length=240)
    trend_source: str = ""
    history: list[SpendYearSchema] = Field(default_factory=list)
    total_csr_inr_crore: float | None = None
    total_csr_display: str = ""
    total_csr_fiscal_year: str = ""
    estimated_min_inr_crore: float | None = None
    estimated_basis: str = Field(default="", max_length=200)
    estimated_is_computed: bool = False


class ProgrammeSchema(BaseModel):
    name: str = ""
    what_is_funded: str = Field(default="", max_length=200)
    beneficiary_group: str = Field(default="", max_length=160)
    beneficiary_type: str = "OTHER"
    description: str = Field(default="", max_length=220)
    is_multi_year: bool = False
    cohort_or_scale: str = ""
    source_excerpt: str = Field(default="", max_length=200)
    source: str = ""
    confidence: str = "confirmed"


class PartnerSchema(BaseModel):
    name: str = ""
    relationship_type: str = ""
    programme: str = Field(default="", max_length=160)
    year: str = ""
    geography: str = Field(default="", max_length=120)
    similar_to_tap_profile: bool = False
    source_excerpt: str = Field(default="", max_length=200)
    source: str = ""
    confidence: str = "confirmed"


class DecisionMakerSchema(BaseModel):
    name: str = ""
    title: str = ""
    public_facing_score: int = Field(ge=0, le=100, default=0)
    tenure_status: str = "UNKNOWN"
    tenure_evidence: str = Field(default="", max_length=200)
    source_excerpt: str = Field(default="", max_length=200)
    source: str = ""
    linkedin_url: str = ""


class GeographySchema(BaseModel):
    place: str = ""
    source_excerpt: str = Field(default="", max_length=160)
    source: str = ""


class RedFlagSchema(BaseModel):
    flag: str = ""
    severity: str = ""
    explanation: str = Field(default="", max_length=220)
    source: str = ""


class ContactPathwaySchema(BaseModel):
    channel: str = ""
    evidence: str = Field(default="", max_length=200)
    source: str = ""


class RfpSignalSchema(BaseModel):
    present: bool = False
    channel: str = ""
    evidence: str = Field(default="", max_length=220)
    source: str = ""


class BoardAffinitySchema(BaseModel):
    present: bool = False
    person_name: str = ""
    connection: str = Field(default="", max_length=220)
    source_excerpt: str = Field(default="", max_length=200)
    source: str = ""


class VolunteeringSchema(BaseModel):
    present: bool = False
    programme_name: str = ""
    description: str = Field(default="", max_length=220)
    source_excerpt: str = Field(default="", max_length=200)
    source: str = ""


class GroupFoundationSchema(BaseModel):
    routed_through_group: bool = False
    foundation_name: str = ""
    explanation: str = Field(default="", max_length=240)
    source_excerpt: str = Field(default="", max_length=200)
    source: str = ""


class EligibilitySchema(BaseModel):
    plausibly_mandated: str = "UNKNOWN"
    reasoning: str = Field(default="", max_length=280)
    net_worth_turnover_signal: str = Field(default="", max_length=200)
    net_worth_turnover_inr_crore: float | None = None
    net_profit_inr_crore: float | None = None
    source: str = ""


class SectorSchema(BaseModel):
    sector: str = "UNKNOWN"
    sub_sector: str = ""
    reasoning: str = Field(default="", max_length=200)


class FullAnalysisSchema(BaseModel):
    fit_score: int = Field(ge=0, le=100, default=0)
    fit_rationale: str = Field(default="", max_length=600)
    overall_semantic_alignment: int = Field(ge=0, le=100, default=0)
    alignment_rationale: str = Field(default="", max_length=500)
    delivery_model: str = "UNCLEAR"
    delivery_model_evidence: str = Field(default="", max_length=220)
    delivery_model_source: str = ""
    spend: SpendSchema = SpendSchema()
    programmes: list[ProgrammeSchema] = Field(default_factory=list)
    partners: list[PartnerSchema] = Field(default_factory=list)
    decision_makers: list[DecisionMakerSchema] = Field(default_factory=list)
    geographies: list[GeographySchema] = Field(default_factory=list)
    criteria: list[CriterionResultSchema] = Field(default_factory=list)
    red_flags: list[RedFlagSchema] = Field(default_factory=list)
    contact_pathway: ContactPathwaySchema = ContactPathwaySchema()
    rfp_signal: RfpSignalSchema = RfpSignalSchema()
    board_affinity: BoardAffinitySchema = BoardAffinitySchema()
    volunteering: VolunteeringSchema = VolunteeringSchema()
    group_foundation: GroupFoundationSchema = GroupFoundationSchema()
    eligibility: EligibilitySchema = EligibilitySchema()
    sector: SectorSchema = SectorSchema()
    evidence_recency: str = Field(default="", max_length=160)
    csr_head_note: str = Field(default="", max_length=320)
    source_quality_assessment: str = Field(default="", max_length=320)
    overall_authenticity_score: int = Field(ge=0, le=100, default=0)
    open_questions: list[str] = Field(default_factory=list)
    strategic_insight: str = Field(default="", max_length=2200)


_cooldown_until_monotonic: float = 0.0
_cooldown_reason: str = ""
_TPM_WINDOW_SECONDS = 60.0
_tpm_window_events: list[tuple] = []
_last_call_finished_at_monotonic: float = 0.0
_anthropic_call_lock = asyncio.Lock()


def _prune_tpm_window(now: float) -> None:
    cutoff = now - _TPM_WINDOW_SECONDS
    while _tpm_window_events and _tpm_window_events[0][0] < cutoff:
        _tpm_window_events.pop(0)


def _record_tpm_usage(tokens: int) -> None:
    now = time.monotonic()
    _prune_tpm_window(now)
    _tpm_window_events.append((now, tokens))


def tpm_tokens_used_in_window() -> int:
    now = time.monotonic()
    _prune_tpm_window(now)
    return sum(tokens for _, tokens in _tpm_window_events)


def tpm_tokens_available(safety_margin: int = 300) -> int:
    used = tpm_tokens_used_in_window()
    return max(0, settings.anthropic_tpm_limit - used - safety_margin)


def _parse_retry_after_seconds(retry_after_header: str, response_body_text: str) -> float:
    try:
        return float(retry_after_header)
    except (TypeError, ValueError):
        pass
    match = re.search(r"try again in ([\d.]+)s", response_body_text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return 30.0


def anthropic_cooldown_remaining_seconds() -> float:
    return max(0.0, _cooldown_until_monotonic - time.monotonic())


def evidence_token_budget(company: str, mission: str, sources_manifest: str, mode: str = "deep") -> int:
    scaffold_tokens = estimate_tokens(
        _extraction_prompt(company, mission, "", sources_manifest)
    ) + SCAFFOLD_SAFETY_MARGIN
    output_reserve = EXTRACTION_OUTPUT_TOKEN_RESERVE
    static_budget = settings.anthropic_tpm_limit - output_reserve - scaffold_tokens
    live_budget = tpm_tokens_available(safety_margin=output_reserve + scaffold_tokens)
    budget = min(static_budget, live_budget) if live_budget > 0 else static_budget
    return max(MIN_EVIDENCE_TOKEN_BUDGET, budget)


async def call_anthropic_chat(
    prompt: str,
    max_tokens: int = 1400,
    temperature: float = 0.0,
    model: str | None = None,
    caller: str = "unknown",
) -> str | None:
    global _cooldown_until_monotonic, _cooldown_reason, _last_call_finished_at_monotonic

    if not settings.anthropic_configured:
        logger.warning("anthropic call skipped caller=%s reason=not_configured", caller)
        return None

    cooldown_remaining = anthropic_cooldown_remaining_seconds()
    if cooldown_remaining > 0:
        logger.warning(
            "anthropic call skipped caller=%s reason=cooldown_active seconds_left=%.0f last_reason=%s",
            caller, cooldown_remaining, _cooldown_reason,
        )
        return None

    async with _anthropic_call_lock:
        since_last_call = time.monotonic() - _last_call_finished_at_monotonic
        if since_last_call < INTER_CALL_DELAY_SECONDS:
            await asyncio.sleep(INTER_CALL_DELAY_SECONDS - since_last_call)

        estimated_prompt_tokens = estimate_tokens(prompt)
        estimated_total_tokens = estimated_prompt_tokens + max_tokens
        hard_ceiling = settings.anthropic_tpm_limit - max_tokens

        if estimated_prompt_tokens > hard_ceiling:
            logger.error(
                "anthropic call aborted before send caller=%s estimated_prompt_tokens=%d max_tokens=%d tpm_limit=%d",
                caller, estimated_prompt_tokens, max_tokens, settings.anthropic_tpm_limit,
            )
            return None

        tokens_used_in_window = tpm_tokens_used_in_window()
        if tokens_used_in_window + estimated_total_tokens > settings.anthropic_tpm_limit:
            logger.warning(
                "anthropic call skipped caller=%s reason=local_tpm_budget_exhausted used=%d estimated=%d limit=%d",
                caller, tokens_used_in_window, estimated_total_tokens, settings.anthropic_tpm_limit,
            )
            _cooldown_until_monotonic = max(_cooldown_until_monotonic, time.monotonic() + 15.0)
            _cooldown_reason = "local tpm budget exhausted"
            return None

        resolved_model = model or settings.anthropic_model
        payload = {
            "model": resolved_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "{"},
            ],
        }

        logger.info(
            "anthropic request caller=%s model=%s max_tokens=%d estimated_prompt_tokens=%d",
            caller, resolved_model, max_tokens, estimated_prompt_tokens,
        )
        request_started_at = time.monotonic()
        _record_tpm_usage(estimated_total_tokens)

        try:
            async with httpx.AsyncClient(timeout=ANTHROPIC_REQUEST_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    ANTHROPIC_MESSAGES_URL,
                    headers={
                        "x-api-key": settings.anthropic_api_key,
                        "anthropic-version": ANTHROPIC_API_VERSION,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
        except httpx.HTTPError as exc:
            logger.error("anthropic transport error caller=%s error=%s", caller, exc)
            _last_call_finished_at_monotonic = time.monotonic()
            return None

        elapsed_ms = (time.monotonic() - request_started_at) * 1000
        _last_call_finished_at_monotonic = time.monotonic()

        if response.status_code == 429:
            retry_after_seconds = _parse_retry_after_seconds(response.headers.get("retry-after", ""), response.text)
            _cooldown_until_monotonic = time.monotonic() + retry_after_seconds
            _cooldown_reason = response.text[:200]
            logger.warning("anthropic 429 caller=%s retry_after=%.0fs", caller, retry_after_seconds)
            return None

        if response.status_code >= 400:
            logger.error("anthropic http error caller=%s status=%d body=%s", caller, response.status_code, response.text[:400])
            return None

        try:
            body = response.json()
        except ValueError:
            logger.error("anthropic non-json response caller=%s", caller)
            return None

        logger.info("anthropic response caller=%s status=%d elapsed_ms=%.0f", caller, response.status_code, elapsed_ms)

        usage = body.get("usage") or {}
        input_tokens, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
        if isinstance(input_tokens, int) and isinstance(output_tokens, int):
            _record_tpm_usage((input_tokens + output_tokens) - estimated_total_tokens)

        if body.get("stop_reason") == "max_tokens":
            logger.warning("anthropic response TRUNCATED caller=%s max_tokens=%d", caller, max_tokens)

        content_blocks = body.get("content") or []
        text_parts = [block.get("text", "") for block in content_blocks if block.get("type") == "text"]
        if not text_parts:
            logger.error("anthropic malformed response caller=%s", caller)
            return None
        return "{" + "".join(text_parts)


def parse_json_response(raw_text: str | None) -> dict:
    if not raw_text:
        return {}
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        pass
    recovered = _recover_partial_json(cleaned)
    if recovered:
        logger.info("parse_json_response recovered via partial-json fallback chars=%d", len(cleaned))
        return recovered
    logger.error("parse_json_response failed to recover any JSON chars=%d", len(cleaned))
    return {}


def _recover_partial_json(cleaned: str, required_key: str | None = None) -> dict:
    decoder = json.JSONDecoder()
    for cut_point in range(len(cleaned), 0, -1):
        candidate = cleaned[:cut_point].rstrip()
        if not candidate:
            continue
        trimmed = candidate.rstrip(",")
        for closers in ("", "}", "]}", "]}}", "}]}", "}]}}"):
            attempt = trimmed + closers
            try:
                parsed = decoder.decode(attempt)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(parsed, dict) and (required_key is None or parsed.get(required_key) is not None):
                return parsed
        if cut_point < len(cleaned) - 4000:
            break
    return {}


_STRAY_MARKER = re.compile(r"\*{3,}")
_DOUBLE_STAR = re.compile(r"\*\*")
_LINKEDIN_PROFILE_URL = re.compile(r"^https?://([a-z]{2,3}\.)?linkedin\.com/in/[^/?#\s]+/?(?:[?#].*)?$", re.IGNORECASE)


def _normalize_highlight_markers(text: str) -> str:
    if not text:
        return text
    cleaned = _STRAY_MARKER.sub("**", text)
    if len(_DOUBLE_STAR.findall(cleaned)) % 2 != 0:
        cleaned = cleaned.replace("**", "")
    return cleaned


def _sanitize_linkedin_url(url: str) -> str:
    cleaned = (url or "").strip()
    return cleaned if _LINKEDIN_PROFILE_URL.match(cleaned) else ""


def _field_max_length(field) -> int | None:
    for constraint in field.metadata:
        if hasattr(constraint, "max_length"):
            return constraint.max_length
    return None


def _field_numeric_bounds(field) -> tuple[float | None, float | None]:
    lower, upper = None, None
    for constraint in field.metadata:
        if hasattr(constraint, "ge"):
            lower = constraint.ge
        if hasattr(constraint, "gt"):
            lower = constraint.gt
        if hasattr(constraint, "le"):
            upper = constraint.le
        if hasattr(constraint, "lt"):
            upper = constraint.lt
    return lower, upper


def _sanitize_value_for_field(value, field):
    annotation = field.annotation
    origin = typing.get_origin(annotation)

    if origin is list:
        if not isinstance(value, list):
            return []
        (item_type,) = typing.get_args(annotation)
        if isinstance(item_type, type) and issubclass(item_type, BaseModel):
            return [_sanitize_dict_for_model(item, item_type) for item in value if isinstance(item, dict)]
        if item_type is str:
            return [str(item)[:2000] for item in value if isinstance(item, str) and item.strip()]
        return value

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _sanitize_dict_for_model(value if isinstance(value, dict) else {}, annotation)

    unwrapped = annotation
    type_args = typing.get_args(annotation)
    if type_args and type(None) in type_args:
        non_none = [a for a in type_args if a is not type(None)]
        unwrapped = non_none[0] if non_none else annotation

    if unwrapped is str:
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        max_length = _field_max_length(field)
        if max_length is not None and len(value) > max_length:
            return value[: max_length - 1].rstrip() + "…" if max_length > 1 else value[:max_length]
        return value

    if unwrapped is bool:
        return bool(value) if value is not None else False

    if unwrapped in (int, float):
        if not isinstance(value, (int, float)):
            return None
        lower, upper = _field_numeric_bounds(field)
        if lower is not None and value < lower:
            value = lower
        if upper is not None and value > upper:
            value = upper
        return int(value) if unwrapped is int else float(value)

    return value


def _sanitize_dict_for_model(data: dict, model: type[BaseModel]) -> dict:
    if not isinstance(data, dict):
        data = {}
    sanitized = {}
    for field_name, field in model.model_fields.items():
        if field_name not in data:
            continue
        sanitized[field_name] = _sanitize_value_for_field(data[field_name], field)
    return sanitized


# --------------------------------------------------------------------------
# DETERMINISTIC FIT-SCORE COMPUTATION
#
# This is the concrete fix for "score doesn't match its own rubric": instead
# of trusting the model's single fit_score number (which the old prompt only
# ever *asked* to be self-consistent with the 17 criteria, with nothing
# enforcing it), the actual number shown to the user is always computed here
# in Python from criteria_score * criteria_weight, scaled 0-5 -> 0-100. The
# model's own fit_score is kept only as a logged cross-check value.
#
# Mode calibration and the authenticity-based ceiling are also both applied
# here deterministically, rather than being just another paragraph the model
# has to remember to obey inside one giant prompt.
# --------------------------------------------------------------------------

def _compute_weighted_fit_score(criteria: list[dict]) -> float:
    if not criteria:
        return 0.0
    total_weight = 0.0
    weighted_sum = 0.0
    for entry in criteria:
        criterion_id = entry.get("id", "")
        weight = CRITERIA_WEIGHTS.get(criterion_id)
        if weight is None:
            continue
        score_0_to_5 = max(0.0, min(5.0, float(entry.get("score", 0) or 0)))
        weighted_sum += (score_0_to_5 / 5.0) * weight
        total_weight += weight
    if total_weight == 0:
        return 0.0
    # total_weight should already be 100 (all 17 present), but normalize
    # defensively in case the model dropped or duplicated an id.
    return (weighted_sum / total_weight) * 100.0


def _apply_authenticity_ceiling(fit_score: float, authenticity_score: int, mode: str) -> float:
    calibration = _mode_calibration(mode)
    threshold = calibration["authenticity_cap_threshold"]
    ceiling = calibration["authenticity_cap_ceiling"]
    if authenticity_score < threshold and fit_score > ceiling:
        return float(ceiling)
    return fit_score


def compute_final_fit_score(criteria: list[dict], authenticity_score: int, mode: str,
                             model_reported_score: int | None = None) -> int:
    weighted = _compute_weighted_fit_score(criteria)
    calibrated = _apply_authenticity_ceiling(weighted, authenticity_score, mode)
    final_score = int(round(max(0.0, min(100.0, calibrated))))
    if model_reported_score is not None:
        drift = abs(model_reported_score - weighted)
        if drift > 15:
            logger.info(
                "fit_score deterministic override: model_reported=%s weighted_from_criteria=%.1f "
                "final=%d mode=%s authenticity=%d drift=%.1f",
                model_reported_score, weighted, final_score, mode, authenticity_score, drift,
            )
    return final_score


def _repair_extraction(parsed: dict) -> dict:
    """Sanitizes the extraction-pass output using the same field-level rules
    as the full analysis schema, but only for the subset of fields the
    extraction pass produces (no criteria, fit_score, or narrative fields —
    those belong to the scoring pass). Reuses FullAnalysisSchema for
    sanitization since the field shapes are identical/overlapping, then
    strips back down to just the extraction fields.
    """
    parsed = dict(parsed) if isinstance(parsed, dict) else {}
    sanitized = _sanitize_dict_for_model(parsed, FullAnalysisSchema)

    if isinstance(parsed.get("decision_makers"), list):
        for entry in sanitized.get("decision_makers", []):
            if isinstance(entry, dict) and entry.get("linkedin_url"):
                entry["linkedin_url"] = _sanitize_linkedin_url(entry["linkedin_url"])

    sanitized["key_facts_summary"] = str(parsed.get("key_facts_summary", "") or "")[:1500]
    sanitized["open_questions"] = [
        str(q).strip()[:200] for q in (parsed.get("open_questions") or []) if q and str(q).strip()
    ][:5]
    return sanitized


def _repair_analysis(parsed: dict) -> FullAnalysisSchema:
    parsed = dict(parsed) if isinstance(parsed, dict) else {}
    parsed = _sanitize_dict_for_model(parsed, FullAnalysisSchema)

    raw_criteria = parsed.get("criteria") if isinstance(parsed.get("criteria"), list) else []
    repaired_criteria, seen_ids = [], set()
    for entry in raw_criteria:
        if not isinstance(entry, dict):
            continue
        criterion_id = entry.get("id")
        if criterion_id not in CRITERIA_IDS or criterion_id in seen_ids:
            continue
        seen_ids.add(criterion_id)
        repaired_criteria.append({
            "id": criterion_id,
            "name": CRITERIA_TITLES[criterion_id],
            "score": min(max(float(entry.get("score", 0) or 0), 0), 5),
            "confidence": int(min(max(entry.get("confidence", 0) or 0, 0), 100)),
            "evidence": str(entry.get("evidence", ""))[:240],
            "reasoning": str(entry.get("reasoning", ""))[:240],
            "source": str(entry.get("source", "")),
        })
    for criterion_id in CRITERIA_IDS:
        if criterion_id not in seen_ids:
            repaired_criteria.append({
                "id": criterion_id, "name": CRITERIA_TITLES[criterion_id],
                "score": 0.0, "confidence": 0,
                "evidence": "No signal returned for this criterion", "reasoning": "", "source": "",
            })
    ordered = {c["id"]: c for c in repaired_criteria}
    parsed["criteria"] = [ordered[cid] for cid in CRITERIA_IDS]

    for field_name in ("fit_rationale", "alignment_rationale", "delivery_model_evidence",
                       "csr_head_note", "evidence_recency", "source_quality_assessment",
                       "strategic_insight"):
        if isinstance(parsed.get(field_name), str):
            parsed[field_name] = _normalize_highlight_markers(parsed[field_name])
    if isinstance(parsed.get("contact_pathway"), dict) and isinstance(parsed["contact_pathway"].get("channel"), str):
        parsed["contact_pathway"]["channel"] = _normalize_highlight_markers(parsed["contact_pathway"]["channel"])

    if isinstance(parsed.get("decision_makers"), list):
        for entry in parsed["decision_makers"]:
            if isinstance(entry, dict) and entry.get("linkedin_url"):
                entry["linkedin_url"] = _sanitize_linkedin_url(entry["linkedin_url"])

    try:
        return FullAnalysisSchema.model_validate(parsed)
    except ValidationError as exc:
        logger.warning("analysis validation failed, repairing containers error=%s", exc)
        for container_field, default in (
            ("spend", {}), ("contact_pathway", {}), ("rfp_signal", {}), ("board_affinity", {}),
            ("volunteering", {}), ("group_foundation", {}), ("eligibility", {}), ("sector", {}),
            ("programmes", []), ("partners", []), ("decision_makers", []), ("geographies", []),
            ("red_flags", []), ("open_questions", []),
        ):
            current = parsed.get(container_field)
            expected_type = list if isinstance(default, list) else dict
            if not isinstance(current, expected_type):
                parsed[container_field] = default
        try:
            return FullAnalysisSchema.model_validate(parsed)
        except ValidationError as exc2:
            logger.error(
                "analysis validation failed even after container repair error=%s — "
                "salvaging each nested item independently rather than discarding the whole analysis",
                exc2,
            )
            safe_kwargs: dict = {
                "fit_score": int(min(max(parsed.get("fit_score", 0) or 0, 0), 100)),
                "criteria": [CriterionResultSchema(**c) for c in repaired_criteria],
            }
            for scalar_field in (
                "fit_rationale", "overall_semantic_alignment", "alignment_rationale",
                "delivery_model", "delivery_model_evidence", "evidence_recency",
                "csr_head_note", "source_quality_assessment", "overall_authenticity_score",
                "strategic_insight",
            ):
                if scalar_field in parsed:
                    safe_kwargs[scalar_field] = parsed[scalar_field]
            for object_field, schema in (
                ("spend", SpendSchema), ("contact_pathway", ContactPathwaySchema),
                ("rfp_signal", RfpSignalSchema), ("board_affinity", BoardAffinitySchema),
                ("volunteering", VolunteeringSchema), ("group_foundation", GroupFoundationSchema),
                ("eligibility", EligibilitySchema), ("sector", SectorSchema),
            ):
                candidate = parsed.get(object_field)
                if isinstance(candidate, dict):
                    try:
                        safe_kwargs[object_field] = schema(**_sanitize_dict_for_model(candidate, schema))
                    except ValidationError:
                        continue
            for list_field, schema in (
                ("programmes", ProgrammeSchema), ("partners", PartnerSchema),
                ("decision_makers", DecisionMakerSchema), ("geographies", GeographySchema),
                ("red_flags", RedFlagSchema),
            ):
                candidates = parsed.get(list_field)
                kept = []
                if isinstance(candidates, list):
                    for item in candidates:
                        if isinstance(item, dict):
                            try:
                                kept.append(schema(**_sanitize_dict_for_model(item, schema)))
                            except ValidationError:
                                continue
                safe_kwargs[list_field] = kept
            if isinstance(parsed.get("open_questions"), list):
                safe_kwargs["open_questions"] = [q for q in parsed["open_questions"] if isinstance(q, str)]
            try:
                return FullAnalysisSchema(**safe_kwargs)
            except ValidationError:
                logger.error("analysis validation failed even after item-level salvage — using minimal fallback")
                return FullAnalysisSchema(
                    fit_score=int(min(max(parsed.get("fit_score", 0) or 0, 0), 100)),
                    criteria=[CriterionResultSchema(**c) for c in repaired_criteria],
                )


def _valid_source_lookup(sources_manifest: str) -> set[str]:
    valid = set()
    for line in sources_manifest.splitlines():
        parts = line.split("|")
        if parts and parts[0].strip():
            valid.add(parts[0].strip())
    return valid


def _sanitize_source(value: str, valid_sources: set[str]) -> str:
    cleaned = (value or "").strip()
    return cleaned if cleaned in valid_sources else ""


def _shrink_to_fit(company: str, mission: str, sources_manifest: str, cleaned_sources: list[dict],
                    prompt_builder, output_ceiling: int) -> tuple[str, list[dict], int]:
    """Shared shrink loop used by both the extraction and scoring passes:
    repeatedly trims FOUND source text proportionally until the built prompt
    fits inside output_ceiling tokens, or gives up after
    MAX_PROMPT_SHRINK_ATTEMPTS. Returns (evidence_text, working_sources,
    final_prompt_tokens); prompt_builder(evidence_text) -> str.
    """
    evidence_text = combine_evidence_text(cleaned_sources)
    prompt = prompt_builder(evidence_text)
    prompt_tokens = estimate_tokens(prompt)
    working_sources = cleaned_sources
    shrink_attempts = 0

    while prompt_tokens > output_ceiling and shrink_attempts < MAX_PROMPT_SHRINK_ATTEMPTS:
        current_evidence_tokens = estimate_tokens(evidence_text)
        if current_evidence_tokens <= 0:
            break
        overflow = prompt_tokens - output_ceiling
        target_evidence_tokens = max(MIN_EVIDENCE_TOKEN_BUDGET, current_evidence_tokens - overflow - PROMPT_SHRINK_SAFETY_MARGIN)
        overflow_ratio = target_evidence_tokens / current_evidence_tokens
        working_sources = [
            {**s, "text": s["text"][: max(MIN_PROMPT_TRIM_CHARS, int(len(s["text"]) * overflow_ratio))]}
            if s.get("status") == "FOUND" else s
            for s in working_sources
        ]
        evidence_text = combine_evidence_text(working_sources)
        prompt = prompt_builder(evidence_text)
        prompt_tokens = estimate_tokens(prompt)
        shrink_attempts += 1

    if shrink_attempts:
        logger.info(
            "shrink_to_fit trimmed evidence company=%r attempts=%d final_prompt_tokens=%d",
            company, shrink_attempts, prompt_tokens,
        )
    return evidence_text, working_sources, prompt_tokens


async def extract_company_facts(
    company: str,
    mission: str,
    cleaned_sources: list[dict],
    sources_manifest: str,
) -> dict | None:
    """Pass 1: pure fact extraction, no scoring. See module docstring above
    _extraction_prompt for why this is split from scoring.
    """
    evidence_text = combine_evidence_text(cleaned_sources)
    if not evidence_text.strip():
        logger.info("extract_company_facts skipped company=%r reason=no_evidence_text", company)
        return None

    output_ceiling = settings.anthropic_tpm_limit - EXTRACTION_OUTPUT_TOKEN_RESERVE

    def _build(evidence: str) -> str:
        return _extraction_prompt(company, mission, evidence, sources_manifest)

    evidence_text, working_sources, prompt_tokens = _shrink_to_fit(
        company, mission, sources_manifest, cleaned_sources, _build, output_ceiling,
    )

    if prompt_tokens > output_ceiling:
        logger.error(
            "extract_company_facts could not fit prompt within budget company=%r prompt_tokens=%d ceiling=%d",
            company, prompt_tokens, output_ceiling,
        )
        return None

    prompt = _build(evidence_text)
    raw_reply = await call_anthropic_chat(
        prompt,
        temperature=0.0,
        max_tokens=EXTRACTION_OUTPUT_TOKEN_RESERVE,
        caller=f"extract_facts:{company}",
    )
    if raw_reply is None:
        logger.error("extract_company_facts got no reply company=%r", company)
        return None

    parsed = parse_json_response(raw_reply)
    if not parsed:
        logger.error("extract_company_facts empty parse company=%r", company)
        return None

    extraction = _repair_extraction(parsed)

    valid_sources = _valid_source_lookup(sources_manifest)
    extraction["delivery_model_source"] = _sanitize_source(extraction.get("delivery_model_source", ""), valid_sources)
    extraction.setdefault("spend", {})
    extraction["spend"]["source"] = _sanitize_source(extraction["spend"].get("source", ""), valid_sources)
    extraction["spend"]["trend_source"] = _sanitize_source(extraction["spend"].get("trend_source", ""), valid_sources)
    for entry in extraction["spend"].get("history", []) or []:
        entry["source"] = _sanitize_source(entry.get("source", ""), valid_sources)
    for programme in extraction.get("programmes", []) or []:
        programme["source"] = _sanitize_source(programme.get("source", ""), valid_sources)
    for partner in extraction.get("partners", []) or []:
        partner["source"] = _sanitize_source(partner.get("source", ""), valid_sources)
    for person in extraction.get("decision_makers", []) or []:
        person["source"] = _sanitize_source(person.get("source", ""), valid_sources)
        person["linkedin_url"] = _sanitize_linkedin_url(person.get("linkedin_url", ""))
    for geography in extraction.get("geographies", []) or []:
        geography["source"] = _sanitize_source(geography.get("source", ""), valid_sources)
    for flag in extraction.get("red_flags", []) or []:
        flag["source"] = _sanitize_source(flag.get("source", ""), valid_sources)
    extraction.setdefault("contact_pathway", {})
    extraction["contact_pathway"]["source"] = _sanitize_source(extraction["contact_pathway"].get("source", ""), valid_sources)
    extraction.setdefault("rfp_signal", {})
    extraction["rfp_signal"]["source"] = _sanitize_source(extraction["rfp_signal"].get("source", ""), valid_sources)
    extraction.setdefault("board_affinity", {})
    extraction["board_affinity"]["source"] = _sanitize_source(extraction["board_affinity"].get("source", ""), valid_sources)
    extraction.setdefault("volunteering", {})
    extraction["volunteering"]["source"] = _sanitize_source(extraction["volunteering"].get("source", ""), valid_sources)
    extraction.setdefault("group_foundation", {})
    extraction["group_foundation"]["source"] = _sanitize_source(extraction["group_foundation"].get("source", ""), valid_sources)
    extraction.setdefault("eligibility", {})
    extraction["eligibility"]["source"] = _sanitize_source(extraction["eligibility"].get("source", ""), valid_sources)

    logger.info(
        "extract_company_facts DONE company=%r authenticity=%d partners=%d programmes=%d decision_makers=%d red_flags=%d",
        company, extraction.get("overall_authenticity_score", 0), len(extraction.get("partners", [])),
        len(extraction.get("programmes", [])), len(extraction.get("decision_makers", [])),
        len(extraction.get("red_flags", [])),
    )
    return extraction


async def score_extracted_facts(
    company: str,
    mission: str,
    mode: str,
    extraction: dict,
    sources_manifest: str,
) -> dict | None:
    """Pass 2: scores the 17 criteria and writes narrative fields against the
    already-extracted fact sheet. fit_score returned here is a cross-check
    only — compute_final_fit_score() in analyze_and_score_company() is what
    actually determines the number shown to the user.
    """
    scoring_facts = {
        k: v for k, v in extraction.items()
        if k not in ("open_questions", "key_facts_summary", "overall_authenticity_score",
                      "evidence_recency", "source_quality_assessment", "csr_head_note")
    }
    prompt = _scoring_prompt(company, mission, mode, scoring_facts, sources_manifest)
    prompt_tokens = estimate_tokens(prompt)
    output_reserve = OUTPUT_TOKEN_RESERVE - EXTRACTION_OUTPUT_TOKEN_RESERVE
    output_ceiling = settings.anthropic_tpm_limit - output_reserve

    if prompt_tokens > output_ceiling:
        # Extraction facts are already distilled, so this should be rare;
        # fall back to trimming the largest text-bearing sub-lists if it happens.
        trimmed_facts = dict(scoring_facts)
        for list_field in ("programmes", "partners", "decision_makers", "geographies", "red_flags"):
            if trimmed_facts.get(list_field):
                trimmed_facts[list_field] = trimmed_facts[list_field][:5]
        prompt = _scoring_prompt(company, mission, mode, trimmed_facts, sources_manifest)
        prompt_tokens = estimate_tokens(prompt)

    if prompt_tokens > output_ceiling:
        logger.error(
            "score_extracted_facts could not fit prompt within budget company=%r prompt_tokens=%d ceiling=%d",
            company, prompt_tokens, output_ceiling,
        )
        return None

    raw_reply = await call_anthropic_chat(
        prompt,
        temperature=0.0,
        max_tokens=output_reserve,
        caller=f"score_facts:{company}",
    )
    if raw_reply is None:
        logger.error("score_extracted_facts got no reply company=%r", company)
        return None

    parsed = parse_json_response(raw_reply)
    if not parsed:
        logger.error("score_extracted_facts empty parse company=%r", company)
        return None

    logger.info(
        "score_extracted_facts DONE company=%r model_reported_fit_score=%s",
        company, parsed.get("fit_score"),
    )
    return parsed


async def analyze_and_score_company(
    company: str,
    mission: str,
    cleaned_sources: list[dict],
    sources_manifest: str,
    mode: str = "deep",
) -> dict | None:
    """Public entry point — signature-compatible with the previous single-shot
    version (mode is new but optional and defaults to the old strict/deep
    behaviour, so any existing caller that doesn't pass mode still works
    unchanged). Internally now runs extraction then scoring, then computes
    fit_score deterministically from the weighted criteria rather than
    trusting either LLM call's own number.
    """
    extraction = await extract_company_facts(company, mission, cleaned_sources, sources_manifest)
    if not extraction:
        return None

    scoring = await score_extracted_facts(company, mission, mode, extraction, sources_manifest)
    if not scoring:
        return None

    merged = dict(extraction)
    merged.pop("key_facts_summary", None)
    merged["criteria"] = scoring.get("criteria", [])
    merged["fit_score"] = scoring.get("fit_score", 0)
    merged["fit_rationale"] = scoring.get("fit_rationale", "")
    merged["overall_semantic_alignment"] = scoring.get("overall_semantic_alignment", 0)
    merged["alignment_rationale"] = scoring.get("alignment_rationale", "")
    merged["strategic_insight"] = scoring.get("strategic_insight", "")

    validated = _repair_analysis(merged)
    result = validated.model_dump()

    model_reported_score = result["fit_score"]
    result["fit_score"] = compute_final_fit_score(
        criteria=result["criteria"],
        authenticity_score=result["overall_authenticity_score"],
        mode=mode,
        model_reported_score=model_reported_score,
    )

    if not result.get("strategic_insight", "").strip():
        result["strategic_insight"] = result.get("fit_rationale", "") or LLM_UNAVAILABLE_EVIDENCE

    result["open_questions"] = [q.strip()[:200] for q in extraction.get("open_questions", []) if q and q.strip()][:5]

    logger.info(
        "analyze_and_score_company DONE company=%r mode=%s model_reported_fit_score=%d final_fit_score=%d "
        "authenticity=%d partners=%d programmes=%d decision_makers=%d",
        company, mode, model_reported_score, result["fit_score"], result["overall_authenticity_score"],
        len(result["partners"]), len(result["programmes"]), len(result["decision_makers"]),
    )
    logger.info(
        "analyze_and_score_company criteria breakdown company=%r %s",
        company, {c["id"]: c["score"] for c in result["criteria"]},
    )

    return result


async def api_health_check() -> dict:
    google_ok = settings.google_search_configured
    if not settings.anthropic_configured:
        anthropic_status = {"ok": False, "model": None, "message": "ANTHROPIC_API_KEY not set — analysis and scoring are unavailable"}
    else:
        reply = await call_anthropic_chat('Reply with JSON: {"status":"ok"}', max_tokens=20, caller="api_health_check")
        if reply:
            anthropic_status = {"ok": True, "model": settings.anthropic_model, "message": f"Claude connected ({settings.anthropic_model}) — full AI analysis active"}
        else:
            anthropic_status = {"ok": False, "model": None, "message": "Anthropic API unreachable — analysis and scoring are unavailable"}
    return {
        "anthropic": anthropic_status,
        "google_search": {
            "configured": google_ok,
            "message": "Google Custom Search configured" if google_ok else "Google Search not configured — using DDGS fallback for all queries",
        },
    }