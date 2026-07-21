import asyncio
import functools
import json
import logging
import re
import time
import typing

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.config import settings
from app.pipeline.textproc import build_token_budgeted_evidence, estimate_tokens, structure_all_sources

logger = logging.getLogger("tap.llm")

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"

LLM_UNAVAILABLE_EVIDENCE = "LLM unavailable — unable to generate evidence"

OUTPUT_TOKEN_RESERVE = 6000
SCAFFOLD_SAFETY_MARGIN = 250
MIN_EVIDENCE_BUDGET = 350
INSIGHT_MAX_TOKENS = 500
ELIGIBILITY_MAX_TOKENS = 400
QUESTION_TRIAGE_MAX_TOKENS = 350
QUESTION_RESOLUTION_MAX_TOKENS = 400
INTER_CALL_DELAY_SECONDS = 2.0

ANTHROPIC_REQUEST_TIMEOUT_SECONDS = 120.0

MAX_OPEN_QUESTIONS_TO_RESOLVE = 3

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

QUESTION_CATEGORY_KEYWORDS = {
    "education_programme": ("education", "stem", "skilling", "skill development", "curriculum", "classroom", "learning"),
    "csr_budget": ("budget", "spend", "expenditure", "crore", "lakh", "percentage-of-profit", "financials"),
    "decision_maker": ("decision-maker", "decision maker", "contact", "head of", "who is the", "csr lead"),
    "ngo_partner": ("ngo", "partner", "implementation", "implementing", "funded"),
    "csr_policy": ("policy", "annual report", "csr report", "disclosure"),
}


class CriterionResultSchema(BaseModel):
    id: str
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
    has_disclosed_budget: bool = False
    confidence: int = Field(ge=0, le=100, default=0)
    source_excerpt: str = Field(default="", max_length=200)
    source: str = ""
    trend_direction: str = "UNKNOWN"
    trend_evidence: str = Field(default="", max_length=240)
    trend_source: str = ""
    history: list[SpendYearSchema] = Field(default_factory=list)
    estimated_min_inr_crore: float | None = None
    estimated_basis: str = Field(default="", max_length=200)
    estimated_is_computed: bool = False


class ProgrammeSchema(BaseModel):
    name: str = ""
    description: str = Field(default="", max_length=220)
    is_multi_year: bool = False
    cohort_or_scale: str = ""
    source_excerpt: str = Field(default="", max_length=200)
    source: str = ""
    confidence: str = "confirmed"


class PartnerSchema(BaseModel):
    name: str = ""
    relationship_type: str = ""
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


class ImportantLinkSchema(BaseModel):
    label: str
    url: str
    relevance: str = Field(default="", max_length=140)


class ImportantLinksSchema(BaseModel):
    links: list[ImportantLinkSchema] = Field(default_factory=list)


class StrategicInsightSchema(BaseModel):
    narrative: str = Field(default="", max_length=2200)


class PersonMatchSchema(BaseModel):
    name: str = ""
    title: str = ""
    is_current_csr_role: bool = False
    match_confidence: int = Field(ge=0, le=100, default=0)
    linkedin_url: str = ""
    tenure_status: str = "UNKNOWN"
    reasoning: str = Field(default="", max_length=180)


class PeopleMatchListSchema(BaseModel):
    people: list[PersonMatchSchema] = Field(default_factory=list)


class QuestionResolutionSchema(BaseModel):
    answered: bool = False
    answer: str = Field(default="", max_length=300)
    confidence: int = Field(ge=0, le=100, default=0)
    updates: dict = Field(default_factory=dict)


def clamp_int(value, minimum: int, maximum: int, default: int) -> int:
    if not isinstance(value, (int, float)):
        return default
    return int(min(max(value, minimum), maximum))


def clamp_float(value, minimum: float, maximum: float, default: float) -> float:
    if not isinstance(value, (int, float)):
        return default
    return round(min(max(float(value), minimum), maximum), 1)


def _nonempty(*values: str) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


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
    remaining = _cooldown_until_monotonic - time.monotonic()
    return max(0.0, remaining)


_RUBRIC = {
    "education_intervention": "hands-on programme not scholarship",
    "stem": "named STEM/coding/robotics/science exposure",
    "tech_21cs": "tech-delivered learning or 21st-c-skills",
    "public_schooling": "explicit government-school work",
    "systems_change": "teacher training, outcomes, scale/policy",
    "programme_depth": "one-off=lower, named multi-year=higher",
    "partnership_quality": "unnamed single-year=lower, named multi-year=higher",
    "decision_maker_accessibility": "named individual with current CSR-decision title",
    "csr_trajectory": "expansion=higher, static=medium, contraction=lower, no signal=0",
    "delivery_model_fit": "how cleanly TAP could enter as grantee or delivery partner",
    "outreach_readiness": "open call/RFP=high, closed programme=low",
    "funding_capacity": "does the disclosed CSR budget plausibly cover a grant of TAP's typical size",
    "csr_spend_trend": "rising multi-year=high, flat=medium, declining=low, no data=0",
    "decision_maker_tenure": "recently appointed=higher signal, entrenched/no signal=lower",
    "group_foundation_routing": "named parent foundation=high, no signal=0",
    "board_education_affinity": "named personal history=higher, generic=low, none=0",
    "employee_volunteering": "active named education programme=higher, generic=low, none=0",
}


def _criteria_rubric_block() -> str:
    return "\n".join(f"- {key}: {value}" for key, value in _RUBRIC.items())


def _criteria_json_template() -> str:
    lines = [
        '    {{"id": "{id}", "score": <0-5>, "confidence": <0-100>, "evidence": "<short>", "reasoning": "<short>"}}'.format(id=cid)
        for cid in CRITERIA_IDS
    ]
    return ",\n".join(lines)


HIGHLIGHT_RULE = (
    "MARKER-HIGHLIGHT RULE (fit_rationale, alignment_rationale, delivery_model_evidence, "
    "source_quality_assessment, csr_head_note, evidence_recency, contact_pathway.channel, "
    "and every criterion evidence field): wrap the single most decision-relevant 2-3 word "
    "phrase in **double asterisks** — never a full sentence, never a lone number, never "
    "more than 3 words. Exactly one bolded phrase per field with real content; zero only "
    "if the field is genuinely empty. Never bold anything in name/title/label/source/url/"
    "boolean/enum fields."
)

OUTPUT_ORDER_RULE = (
    "OUTPUT ORDER RULE: write fit_score, fit_rationale, overall_semantic_alignment, "
    "alignment_rationale, delivery_model, delivery_model_evidence as the first six keys, "
    "in that order. Everything after (spend, programmes, criteria, etc.) may be trimmed "
    "before those lead fields ever are."
)

SPEND_VS_REVENUE_RULE = (
    "SPEND-VS-REVENUE RULE — apply strictly, this is the most common source of error: "
    "revenue, turnover, net worth, net profit, market cap, and EBITDA are business-scale "
    "figures, NOT CSR spend, and must NEVER populate spend.display, spend.inr_crore, or "
    "be described anywhere as 'CSR spend', 'CSR budget', or 'CSR fund'. Set "
    "spend.has_disclosed_budget true ONLY when the evidence contains a figure explicitly "
    "labeled as CSR expenditure, CSR spend, amount spent on CSR, CSR budget, or a stated "
    "CSR-mandate percentage applied to a stated profit figure. If the evidence contains "
    "only revenue/turnover/net worth/net profit with no explicit CSR-labeled figure, set "
    "spend.has_disclosed_budget false, leave spend.inr_crore null, and instead report the "
    "clean numeric business-scale figures in eligibility.net_worth_turnover_signal (as text) "
    "AND, whichever of net worth/turnover is stated, in eligibility.net_worth_turnover_inr_crore "
    "and eligibility.net_profit_inr_crore (as plain numbers in INR crore, null if not stated) — "
    "this lets a transparent statutory-minimum estimate be computed in code, never by you, and "
    "never written into spend. Anywhere a business-scale figure is mentioned in prose "
    "(fit_rationale, alignment_rationale, csr_head_note), it must be labeled 'revenue'/"
    "'turnover'/'net worth' exactly as the evidence states — never implied to be CSR capacity."
)

PARTNER_INCLUSION_RULE = (
    "PARTNER INCLUSION RULE: the partners array is the source of truth for organisations "
    "the company works with, split into two confidence tiers. confidence='confirmed': the "
    "evidence explicitly states a working relationship (funds, co-designs, implements with, "
    "partners with, delivers via) with a named third-party organisation. confidence='probable': "
    "a named third-party organisation is mentioned alongside the company in a CSR/education "
    "context but the relationship verb is vague or implied rather than explicit. Do NOT add an "
    "entry, at either tier, for: a programme, platform, or campaign name that is not itself a "
    "separate organisation (an internal initiative name is a programme, not a partner); a "
    "government body mentioned only in generic language ('works with governments and NGOs') "
    "with no organisation actually named; an award, index, or certifying body the company "
    "received recognition from. MANDATORY CROSS-CHECK: before finalizing your answer, re-read "
    "every narrative field you are about to write (fit_rationale, alignment_rationale, "
    "delivery_model_evidence, csr_head_note) — if any of them names a specific third-party "
    "organisation the company works with (e.g. 'partnership with AIFT'), that organisation "
    "MUST also appear in the partners array at confirmed or probable tier; it is a contradiction "
    "to describe a partnership in prose while leaving the structured array empty. The only "
    "reason a named organisation from your narrative may be absent from partners is if it is "
    "itself the NGO mission-holder (TAP) or a government body with no organisation name attached. "
    "Each qualifying partner appears exactly once, at its highest-supported tier, using its most "
    "complete verbatim name."
)

PROGRAMME_INCLUSION_RULE = (
    "PROGRAMME INCLUSION RULE: add a programmes entry, tagged confidence='confirmed', only "
    "for a named initiative the company itself runs or funds with at least one concrete "
    "supporting detail (what it does, who it serves, scale, or since when) beyond just a "
    "name. If a programme is named but the only supporting detail is thin or partially "
    "implied, tag it confidence='probable' instead of omitting it or inventing detail to "
    "reach the confirmed bar. A bare name with zero supporting detail anywhere in the "
    "evidence should not receive its own programmes entry at any tier. MANDATORY CROSS-CHECK: "
    "if fit_rationale, alignment_rationale, or delivery_model_evidence names a specific "
    "initiative (e.g. 'the AIFT partnership focuses on coding and digital literacy'), that "
    "initiative MUST also appear in the programmes array at confirmed or probable tier — do "
    "not describe a named initiative in prose while leaving programmes empty."
)

EVIDENCE_ONLY_RULE = (
    "EVIDENCE-ONLY RULE: every structured field and every narrative sentence must trace "
    "to something actually stated in the evidence. Do not infer facts from a company's "
    "sector, size, or general reputation. Where evidence is partial, say so explicitly "
    "(e.g. 'no explicit government-school partnership named') rather than writing the "
    "claim as confirmed. An accurate 0/UNKNOWN/empty value is always preferable to a "
    "plausible-sounding guess. A name or figure earns a place in a structured field only "
    "by satisfying the PARTNER INCLUSION RULE, PROGRAMME INCLUSION RULE, or SPEND-VS-"
    "REVENUE RULE above — mentioning something in prose does not, by itself, qualify it "
    "for the structured array. Generic sector-wide statistics (e.g. 'X% of companies in "
    "this sector partner with NGOs') are never evidence of this specific company's "
    "activity and must not be cited to support any criterion score or structured field."
)


def full_company_analysis_prompt(company: str, mission: str, evidence_text: str, sources_manifest: str) -> str:
    return f"""You are a careful, skeptical CSR partnerships analyst judging whether {company} is a genuinely good funding/partnership fit for an Indian education NGO. Read the evidence below and form a judgment grounded strictly in what it states. Accuracy matters far more than completeness — an unfilled field is correct if the evidence does not support one; a filled field that goes beyond the evidence is a failure.

NGO MISSION: {mission}

EVIDENCE:
\"\"\"
{evidence_text}
\"\"\"

{OUTPUT_ORDER_RULE}

{SPEND_VS_REVENUE_RULE}

{PARTNER_INCLUSION_RULE}

{PROGRAMME_INCLUSION_RULE}

{EVIDENCE_ONLY_RULE}

{HIGHLIGHT_RULE}

1. FIT SCORE 0-100: grounded in evidence actually present, not sector reputation. A named programme with concrete detail directly touching STEM/technology/21st-century skills and education justifies 60-85 depending on strength and depth of detail. Sector plausibility alone with no named programme justifies at most 35-50. Thin or missing evidence should pull the score down. Reserve above 85 for a named, detailed, multi-year programme plus a disclosed CSR figure plus an identifiable contact path.
2. FIT RATIONALE (REQUIRED, 2-4 sentences): explain fit_score citing only retrieved evidence. State plainly which parts are confirmed versus inferred or missing. Never present revenue/turnover as CSR capacity without labeling it as such.
3. SEMANTIC ALIGNMENT 0-100 (REQUIRED) + ALIGNMENT RATIONALE (REQUIRED, 1-2 sentences), based only on named programme content.
4. DELIVERY MODEL: FUNDER/IMPLEMENTER/HYBRID/UNCLEAR + DELIVERY MODEL EVIDENCE (REQUIRED sentence naming the specific programme/statement; UNCLEAR with empty evidence if the text gives no clue).
5. BUDGET — apply the SPEND-VS-REVENUE RULE above without exception. Latest figure with fiscal_year if stated else null/conf 0; prior years into history[]; trend_direction from actual CSR-labeled numbers only, never from revenue growth. Populate eligibility.net_worth_turnover_inr_crore / eligibility.net_profit_inr_crore whenever those business-scale numbers are stated, even though they never populate spend.
6. PROGRAMMES — apply the PROGRAMME INCLUSION RULE above, including its mandatory cross-check against your own narrative fields; tag confidence confirmed/probable.
7. PARTNERS — apply the PARTNER INCLUSION RULE above, including its mandatory cross-check against your own narrative fields; tag confidence confirmed/probable. A shorter list than the narrative implies is only correct if the narrative itself names no specific organisation — if it does, that organisation belongs in this array.
8. DECISION MAKERS: every named leader/executive/spokesperson quoted or mentioned in a CSR/sustainability context, title, public_facing_score 0-100, tenure_status, linkedin_url only if a literal linkedin.com/in/ URL is present for that person, else empty. Anyone named in contact_pathway MUST appear here too.
9. GEOGRAPHY: every state/city explicitly named in the evidence.
10. RFP SIGNAL: explicit call for NGO partners — present, channel, evidence. Default false/empty unless explicitly stated.
11. BOARD AFFINITY: named board/promoter personal education-philanthropy history. Default false/empty unless explicitly stated.
12. VOLUNTEERING: named employee volunteering/payroll-giving touching education. Default false/empty unless explicitly stated.
13. GROUP FOUNDATION: CSR run via separate parent/group foundation, only if explicitly named.
14. ELIGIBILITY: from net worth/turnover/profit figures (kept separate from spend), Section 135 applicability LIKELY/UNLIKELY/UNKNOWN, plus the plain numeric fields described in rule 5.
15. SECTOR (UNKNOWN only if evidence gives no industry clue): classify from company-description language, sub_sector if clear, one-line reasoning.
16. CRITERIA 0-5 each, all ids in order, short evidence+reasoning: {_criteria_rubric_block()}
17. RED FLAGS: genuine contradictions, marketing-not-substance signals, date mismatches, or internal conflicts with your own output elsewhere. Severity low/medium/high. Unconfirmed details go in open_questions, not red_flags.
18. CONTACT PATHWAY (name the single most concrete real channel; "Not identified" if truly nothing exists — never invent a channel from a generic mention).
19. EVIDENCE RECENCY (one sentence): how recent/current the evidence appears.
20. CSR HEAD NOTE (one sentence): only from actual decision-maker quotes or named structure; do not speculate about philosophy from a bare title.
21. SOURCE QUALITY ASSESSMENT (1-2 sentences): state plainly whether sources were primary (company/regulator) or secondary (press/search snippets), and whether figures used are self-reported vs independently verified.
22. AUTHENTICITY SCORE 0-100: reflect actual sourcing quality — lower it when the only support is a press mention or search snippet rather than a primary document.
23. OPEN QUESTIONS: up to 5 short items to verify, including any figure excluded from spend under the SPEND-VS-REVENUE RULE. Phrase each as a concrete, searchable question (e.g. "Does {company} run a named education or STEM programme in India?" rather than "More detail needed").

All criteria ids below must appear exactly once, in order. Missing evidence for a criterion: score 0, confidence 0, evidence "To confirm — no signal in evidence".

Rules: evidence fields are paraphrases under 20 words, never verbatim except exact figures and exact partner/programme names. Never fabricate facts. Numbers internally consistent. Keep every string concise so the full reply fits within {OUTPUT_TOKEN_RESERVE} output tokens, prioritizing the first six keys per the OUTPUT ORDER RULE. Reply with ONE JSON object, nothing else.

JSON shape:
{{
  "fit_score": <int 0-100>,
  "fit_rationale": "<2-4 sentences, required, one **2-3 word** highlight>",
  "overall_semantic_alignment": <int 0-100>,
  "alignment_rationale": "<1-2 sentences, required, one **2-3 word** highlight>",
  "delivery_model": "<FUNDER|IMPLEMENTER|HYBRID|UNCLEAR>",
  "delivery_model_evidence": "<sentence, required unless truly no clue, one **2-3 word** highlight>",
  "spend": {{"inr_crore": <number or null>, "display": "<exact CSR-labeled figure/unit as stated, never revenue>", "fiscal_year": "<if stated>", "has_disclosed_budget": <bool>, "confidence": <0-100>, "source_excerpt": "<short>", "trend_direction": "<RISING|FLAT|DECLINING|UNKNOWN>", "trend_evidence": "<short>", "history": [{{"fiscal_year": "<year>", "inr_crore": <number or null>, "display": "<as stated>", "source_excerpt": "<short>"}}]}},
  "programmes": [{{"name": "<exact name>", "description": "<short, must include a concrete supporting detail>", "is_multi_year": <bool>, "cohort_or_scale": "<if stated>", "source_excerpt": "<short>", "confidence": "<confirmed|probable>"}}],
  "partners": [{{"name": "<exact standalone organisation name>", "relationship_type": "<funder|implementer|co-design|unclear>", "source_excerpt": "<short, must show relationship language>", "confidence": "<confirmed|probable>"}}],
  "decision_makers": [{{"name": "<name>", "title": "<title>", "public_facing_score": <0-100>, "tenure_status": "<NEW_UNDER_1YR|ESTABLISHED_1_3YR|ENTRENCHED_3YR_PLUS|UNKNOWN>", "tenure_evidence": "<short>", "source_excerpt": "<short>", "linkedin_url": "<url or empty>"}}],
  "geographies": [{{"place": "<place>", "source_excerpt": "<short>"}}],
  "criteria": [
{_criteria_json_template()}
  ],
  "red_flags": [{{"flag": "<short label>", "severity": "<low|medium|high>", "explanation": "<short>"}}],
  "contact_pathway": {{"channel": "<sentence, required, one **2-3 word** highlight>", "evidence": "<short>"}},
  "rfp_signal": {{"present": <bool>, "channel": "<short>", "evidence": "<short>"}},
  "board_affinity": {{"present": <bool>, "person_name": "<name or empty>", "connection": "<short>", "source_excerpt": "<short>"}},
  "volunteering": {{"present": <bool>, "programme_name": "<name or empty>", "description": "<short>", "source_excerpt": "<short>"}},
  "group_foundation": {{"routed_through_group": <bool>, "foundation_name": "<name or empty>", "explanation": "<short>", "source_excerpt": "<short>"}},
  "eligibility": {{"plausibly_mandated": "<LIKELY|UNLIKELY|UNKNOWN>", "reasoning": "<short>", "net_worth_turnover_signal": "<short>", "net_worth_turnover_inr_crore": <number or null>, "net_profit_inr_crore": <number or null>}},
  "sector": {{"sector": "<sector, required>", "sub_sector": "<sub-sector or empty>", "reasoning": "<short, required>"}},
  "evidence_recency": "<one sentence, required, one **2-3 word** highlight>",
  "csr_head_note": "<one sentence, required, one **2-3 word** highlight>",
  "source_quality_assessment": "<1-2 sentences, required, one **2-3 word** highlight>",
  "overall_authenticity_score": <int 0-100>,
  "open_questions": ["<short item>", "..."]
}}"""


@functools.lru_cache(maxsize=8)
def _measured_scaffold_tokens(company_len_bucket: int, mission_len_bucket: int, manifest_len_bucket: int) -> int:
    placeholder_company = "X" * company_len_bucket
    placeholder_mission = "X" * mission_len_bucket
    placeholder_manifest = "X" * manifest_len_bucket
    empty_prompt = full_company_analysis_prompt(placeholder_company, placeholder_mission, "", placeholder_manifest)
    return estimate_tokens(empty_prompt)


def _bucket(length: int, size: int = 64) -> int:
    return ((length // size) + 1) * size


def prompt_scaffold_tokens(company: str, mission: str, sources_manifest: str) -> int:
    return _measured_scaffold_tokens(
        _bucket(len(company)), _bucket(len(mission)), _bucket(len(sources_manifest))
    )


def analysis_input_token_budget(company: str = "", mission: str = "", sources_manifest: str = "") -> int:
    mission = mission or DEFAULT_MISSION
    scaffold_tokens = prompt_scaffold_tokens(company, mission, sources_manifest) + SCAFFOLD_SAFETY_MARGIN
    static_budget = settings.anthropic_tpm_limit - OUTPUT_TOKEN_RESERVE - scaffold_tokens
    live_budget = tpm_tokens_available(safety_margin=OUTPUT_TOKEN_RESERVE + scaffold_tokens)
    budget = min(static_budget, live_budget) if live_budget > 0 else static_budget
    final_budget = max(MIN_EVIDENCE_BUDGET, budget)
    logger.info(
        "token budget scaffold=%d static=%d live=%d final=%d",
        scaffold_tokens, static_budget, live_budget, final_budget,
    )
    return final_budget


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
            wait_seconds = INTER_CALL_DELAY_SECONDS - since_last_call
            logger.info("anthropic pacing delay caller=%s waiting=%.1fs", caller, wait_seconds)
            await asyncio.sleep(wait_seconds)

        estimated_prompt_tokens = estimate_tokens(prompt)
        estimated_total_tokens = estimated_prompt_tokens + max_tokens

        hard_ceiling = settings.anthropic_tpm_limit - max_tokens
        if estimated_prompt_tokens > hard_ceiling:
            logger.error(
                "anthropic call aborted before send caller=%s estimated_prompt_tokens=%d max_tokens=%d tpm_limit=%d hard_ceiling=%d",
                caller, estimated_prompt_tokens, max_tokens, settings.anthropic_tpm_limit, hard_ceiling,
            )
            return None

        tokens_used_in_window = tpm_tokens_used_in_window()
        if tokens_used_in_window + estimated_total_tokens > settings.anthropic_tpm_limit:
            window_wait_hint = max(0.0, _TPM_WINDOW_SECONDS - 1.0)
            logger.warning(
                "anthropic call skipped caller=%s reason=local_tpm_budget_exhausted used_this_window=%d "
                "estimated_total_tokens=%d tpm_limit=%d — would exceed limit, not sending",
                caller, tokens_used_in_window, estimated_total_tokens, settings.anthropic_tpm_limit,
            )
            _cooldown_until_monotonic = max(_cooldown_until_monotonic, time.monotonic() + min(window_wait_hint, 15.0))
            _cooldown_reason = "local tpm budget exhausted, avoided sending a call likely to 429"
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
            "anthropic request caller=%s model=%s max_tokens=%d prompt_chars=%d estimated_prompt_tokens=%d window_used_before=%d timeout_s=%.0f",
            caller, resolved_model, max_tokens, len(prompt), estimated_prompt_tokens, tokens_used_in_window,
            ANTHROPIC_REQUEST_TIMEOUT_SECONDS,
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
            elapsed_ms = (time.monotonic() - request_started_at) * 1000
            logger.error(
                "anthropic transport error caller=%s exc_type=%s elapsed_ms=%.0f timeout_s=%.0f error=%s",
                caller, type(exc).__name__, elapsed_ms, ANTHROPIC_REQUEST_TIMEOUT_SECONDS, exc,
            )
            _last_call_finished_at_monotonic = time.monotonic()
            return None

        elapsed_ms = (time.monotonic() - request_started_at) * 1000
        _last_call_finished_at_monotonic = time.monotonic()

        if response.status_code == 429:
            retry_after_header = response.headers.get("retry-after", "")
            rate_limit_remaining = response.headers.get("anthropic-ratelimit-requests-remaining", "unknown")
            rate_limit_reset = response.headers.get("anthropic-ratelimit-requests-reset", "unknown")
            retry_after_seconds = _parse_retry_after_seconds(retry_after_header, response.text)
            _cooldown_until_monotonic = time.monotonic() + retry_after_seconds
            _cooldown_reason = response.text[:200]
            logger.warning(
                "anthropic 429 RATE LIMITED caller=%s model=%s retry_after=%.0fs remaining_requests=%s reset=%s body=%s",
                caller, resolved_model, retry_after_seconds, rate_limit_remaining, rate_limit_reset, response.text[:400],
            )
            return None

        if response.status_code == 413:
            logger.error(
                "anthropic 413 TOO LARGE caller=%s estimated_prompt_tokens=%d body=%s",
                caller, estimated_prompt_tokens, response.text[:400],
            )
            return None

        if response.status_code >= 400:
            logger.error(
                "anthropic http error caller=%s status=%d elapsed_ms=%.0f body=%s",
                caller, response.status_code, elapsed_ms, response.text[:400],
            )
            return None

        try:
            body = response.json()
        except ValueError:
            logger.error("anthropic non-json response caller=%s status=%d", caller, response.status_code)
            return None

        logger.info("anthropic response caller=%s status=%d elapsed_ms=%.0f", caller, response.status_code, elapsed_ms)

        usage = body.get("usage") or {}
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if isinstance(input_tokens, int) and isinstance(output_tokens, int):
            actual_total_tokens = input_tokens + output_tokens
            _record_tpm_usage(actual_total_tokens - estimated_total_tokens)
            logger.info(
                "anthropic usage caller=%s input_tokens=%d output_tokens=%d total=%d estimated_total=%d delta=%d",
                caller, input_tokens, output_tokens, actual_total_tokens, estimated_total_tokens,
                actual_total_tokens - estimated_total_tokens,
            )

        stop_reason = body.get("stop_reason", "")
        if stop_reason == "max_tokens":
            logger.warning(
                "anthropic response TRUNCATED caller=%s max_tokens=%d — model ran out of output budget, "
                "attempting partial-JSON recovery since lead fields are front-loaded",
                caller, max_tokens,
            )

        content_blocks = body.get("content") or []
        text_parts = [block.get("text", "") for block in content_blocks if block.get("type") == "text"]
        if not text_parts:
            logger.error("anthropic malformed response caller=%s body_keys=%s", caller, list(body.keys()))
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
    end = max(cleaned.rfind("}"), cleaned.rfind("]"))
    if end == -1:
        logger.error("parse_json_response failed to recover any JSON chars=%d", len(cleaned))
        return {}
    for start_offset in range(0, 3):
        try:
            parsed = json.loads(cleaned[: end + 1 - start_offset] + "}" * start_offset)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            continue
    logger.error("parse_json_response exhausted all recovery attempts chars=%d", len(cleaned))
    return {}


def _recover_partial_json(cleaned: str) -> dict:
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
            if isinstance(parsed, dict) and parsed.get("fit_score") is not None:
                return parsed
        if cut_point < len(cleaned) - 4000:
            break
    return {}


_STRAY_MARKER_PATTERN = re.compile(r"\*{3,}")
_UNPAIRED_DOUBLE_STAR_PATTERN = re.compile(r"\*\*")
_LINKEDIN_PROFILE_URL_PATTERN = re.compile(r"^https?://([a-z]{2,3}\.)?linkedin\.com/in/[^/?#\s]+/?(?:[?#].*)?$", re.IGNORECASE)


def _normalize_highlight_markers(text: str) -> str:
    if not text:
        return text
    cleaned = _STRAY_MARKER_PATTERN.sub("**", text)
    if len(_UNPAIRED_DOUBLE_STAR_PATTERN.findall(cleaned)) % 2 != 0:
        cleaned = cleaned.replace("**", "")
    return cleaned


def _sanitize_linkedin_url(url: str) -> str:
    cleaned = (url or "").strip()
    if not cleaned:
        return ""
    return cleaned if _LINKEDIN_PROFILE_URL_PATTERN.match(cleaned) else ""


_REVENUE_LANGUAGE_PATTERN = re.compile(
    r"\b(revenue|turnover|net\s*worth|net\s*profit|ebitda|market\s*cap)\b", re.IGNORECASE
)
_CSR_LABEL_PATTERN = re.compile(
    r"\b(csr\s*(spend|expenditure|budget|fund|obligation)|amount\s*spent\s*(on\s*)?csr|"
    r"csr\s*mandate)\b", re.IGNORECASE
)

STATUTORY_CSR_MIN_PERCENT = 2.0


def _spend_display_is_revenue_like(display: str) -> bool:
    if not display:
        return False
    has_revenue_language = bool(_REVENUE_LANGUAGE_PATTERN.search(display))
    has_csr_label = bool(_CSR_LABEL_PATTERN.search(display))
    return has_revenue_language and not has_csr_label


def _compute_statutory_estimate(eligibility: dict) -> tuple[float | None, str]:
    net_profit = eligibility.get("net_profit_inr_crore")
    if isinstance(net_profit, (int, float)) and net_profit > 0:
        estimate = round(net_profit * STATUTORY_CSR_MIN_PERCENT / 100, 2)
        basis = (
            f"Estimated statutory minimum — not a disclosed figure. Computed as "
            f"{STATUTORY_CSR_MIN_PERCENT:.0f}% of the disclosed net profit "
            f"(₹{net_profit:g} crore) under Section 135, before verification."
        )
        return estimate, basis
    return None, ""


def _enforce_spend_integrity(spend: dict, eligibility: dict) -> dict:
    display = spend.get("display", "") or ""
    excerpt = spend.get("source_excerpt", "") or ""
    if spend.get("has_disclosed_budget") and _spend_display_is_revenue_like(display + " " + excerpt):
        logger.warning(
            "spend integrity guard fired: figure looks like revenue/turnover, not CSR spend — "
            "forcing has_disclosed_budget=false display=%r",
            display,
        )
        spend["has_disclosed_budget"] = False
        spend["inr_crore"] = None
        spend["confidence"] = 0
        spend["display"] = ""

    if not spend.get("has_disclosed_budget"):
        estimate, basis = _compute_statutory_estimate(eligibility)
        if estimate is not None:
            spend["estimated_min_inr_crore"] = estimate
            spend["estimated_basis"] = basis
            spend["estimated_is_computed"] = True
        else:
            spend["estimated_min_inr_crore"] = None
            spend["estimated_basis"] = ""
            spend["estimated_is_computed"] = False
    else:
        spend["estimated_min_inr_crore"] = None
        spend["estimated_basis"] = ""
        spend["estimated_is_computed"] = False
    return spend


_RELATIONSHIP_SIGNAL_PATTERN = re.compile(
    r"\b(fund(s|ed|ing)?|co-design|co-develop|implement(s|ing|ation)?|partner(s|ed|ship)?|"
    r"collaborat|deliver(s|ed|ing)?\s+(via|through|with)|works?\s+with|grant(s|ee|ed)?)\b",
    re.IGNORECASE,
)
_WEAK_MENTION_SIGNAL_PATTERN = re.compile(
    r"\b(alongside|also named|mentioned with|in association|joint|together with|"
    r"as part of|among the)\b",
    re.IGNORECASE,
)
_GENERIC_PARTNER_NAME_PATTERN = re.compile(
    r"^(governments?|ngos?|partners?|the\s+government|state\s+governments?|local\s+"
    r"governments?)$",
    re.IGNORECASE,
)


def _classify_partner_tier(entry: dict) -> str | None:
    name = (entry.get("name") or "").strip()
    if not name or len(name) < 3:
        return None
    if _GENERIC_PARTNER_NAME_PATTERN.match(name):
        return None
    excerpt = entry.get("source_excerpt", "") or ""
    relationship_type = (entry.get("relationship_type") or "").strip().lower()
    stated_confidence = (entry.get("confidence") or "").strip().lower()
    if relationship_type in {"funder", "implementer", "co-design"} or _RELATIONSHIP_SIGNAL_PATTERN.search(excerpt):
        return "confirmed"
    if stated_confidence == "probable" or _WEAK_MENTION_SIGNAL_PATTERN.search(excerpt) or excerpt.strip():
        return "probable"
    return None


def _classify_programme_tier(entry: dict) -> str | None:
    name = (entry.get("name") or "").strip()
    description = (entry.get("description") or "").strip()
    if not name or len(name) < 3:
        return None
    stated_confidence = (entry.get("confidence") or "").strip().lower()
    if len(description) >= 15:
        return "confirmed"
    if description or stated_confidence == "probable":
        return "probable"
    return None


def _field_max_length(field) -> int | None:
    for constraint in field.metadata:
        if hasattr(constraint, "max_length"):
            return constraint.max_length
    return None


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
        return value if isinstance(value, (int, float)) else None

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


def _repair_full_analysis(parsed: dict) -> FullAnalysisSchema:
    if not isinstance(parsed, dict):
        parsed = {}
    parsed = _sanitize_dict_for_model(parsed, FullAnalysisSchema)
    raw_criteria = parsed.get("criteria")
    by_id = {}
    if isinstance(raw_criteria, list):
        for entry in raw_criteria:
            if isinstance(entry, dict) and entry.get("id") in CRITERIA_IDS:
                by_id[entry["id"]] = entry

    repaired_criteria = []
    for criterion_id in CRITERIA_IDS:
        entry = by_id.get(criterion_id, {})
        repaired_criteria.append({
            "id": criterion_id,
            "score": clamp_float(entry.get("score"), 0, 5, 0.0),
            "confidence": clamp_int(entry.get("confidence"), 0, 100, 0),
            "evidence": _normalize_highlight_markers((entry.get("evidence") or "To confirm — no signal returned by model")[:240]),
            "reasoning": (entry.get("reasoning") or "")[:240],
            "source": entry.get("source") or "",
        })
    parsed = dict(parsed)
    parsed["criteria"] = repaired_criteria
    parsed["fit_score"] = clamp_int(parsed.get("fit_score"), 0, 100, 0)

    eligibility_raw = parsed.get("eligibility") if isinstance(parsed.get("eligibility"), dict) else {}

    if isinstance(parsed.get("spend"), dict):
        parsed["spend"] = _enforce_spend_integrity(dict(parsed["spend"]), eligibility_raw)
    else:
        estimate, basis = _compute_statutory_estimate(eligibility_raw)
        parsed["spend"] = {
            "estimated_min_inr_crore": estimate,
            "estimated_basis": basis,
            "estimated_is_computed": estimate is not None,
        }

    if isinstance(parsed.get("partners"), list):
        tiered_partners = []
        dropped = 0
        for entry in parsed["partners"]:
            if not isinstance(entry, dict):
                continue
            tier = _classify_partner_tier(entry)
            if tier is None:
                dropped += 1
                continue
            entry = dict(entry)
            entry["confidence"] = tier
            tiered_partners.append(entry)
        if dropped:
            logger.info("partner integrity guard dropped %d uncredible partner entries", dropped)
        parsed["partners"] = tiered_partners

    if isinstance(parsed.get("programmes"), list):
        tiered_programmes = []
        dropped = 0
        for entry in parsed["programmes"]:
            if not isinstance(entry, dict):
                continue
            tier = _classify_programme_tier(entry)
            if tier is None:
                dropped += 1
                continue
            entry = dict(entry)
            entry["confidence"] = tier
            tiered_programmes.append(entry)
        if dropped:
            logger.info("programme integrity guard dropped %d thin programme entries", dropped)
        parsed["programmes"] = tiered_programmes

    if isinstance(parsed.get("decision_makers"), list):
        cleaned_decision_makers = []
        for entry in parsed["decision_makers"]:
            if not isinstance(entry, dict):
                continue
            entry = dict(entry)
            entry["linkedin_url"] = _sanitize_linkedin_url(entry.get("linkedin_url", ""))
            cleaned_decision_makers.append(entry)
        parsed["decision_makers"] = cleaned_decision_makers

    for narrative_field in (
        "fit_rationale", "alignment_rationale", "delivery_model_evidence",
        "source_quality_assessment", "csr_head_note", "evidence_recency",
    ):
        if isinstance(parsed.get(narrative_field), str):
            parsed[narrative_field] = _normalize_highlight_markers(parsed[narrative_field])
    if isinstance(parsed.get("contact_pathway"), dict) and isinstance(parsed["contact_pathway"].get("channel"), str):
        parsed["contact_pathway"]["channel"] = _normalize_highlight_markers(parsed["contact_pathway"]["channel"])

    try:
        return FullAnalysisSchema.model_validate(parsed)
    except ValidationError as exc:
        logger.warning("full analysis validation failed, repairing containers error=%s", exc)
        parsed["spend"] = parsed.get("spend") if isinstance(parsed.get("spend"), dict) else {}
        parsed["contact_pathway"] = parsed.get("contact_pathway") if isinstance(parsed.get("contact_pathway"), dict) else {}
        parsed["rfp_signal"] = parsed.get("rfp_signal") if isinstance(parsed.get("rfp_signal"), dict) else {}
        parsed["board_affinity"] = parsed.get("board_affinity") if isinstance(parsed.get("board_affinity"), dict) else {}
        parsed["volunteering"] = parsed.get("volunteering") if isinstance(parsed.get("volunteering"), dict) else {}
        parsed["group_foundation"] = parsed.get("group_foundation") if isinstance(parsed.get("group_foundation"), dict) else {}
        parsed["eligibility"] = parsed.get("eligibility") if isinstance(parsed.get("eligibility"), dict) else {}
        parsed["sector"] = parsed.get("sector") if isinstance(parsed.get("sector"), dict) else {}
        parsed["programmes"] = parsed.get("programmes") if isinstance(parsed.get("programmes"), list) else []
        parsed["partners"] = parsed.get("partners") if isinstance(parsed.get("partners"), list) else []
        parsed["decision_makers"] = parsed.get("decision_makers") if isinstance(parsed.get("decision_makers"), list) else []
        parsed["geographies"] = parsed.get("geographies") if isinstance(parsed.get("geographies"), list) else []
        parsed["red_flags"] = parsed.get("red_flags") if isinstance(parsed.get("red_flags"), list) else []
        parsed["open_questions"] = parsed.get("open_questions") if isinstance(parsed.get("open_questions"), list) else []
        try:
            return FullAnalysisSchema.model_validate(parsed)
        except ValidationError as exc2:
            logger.error("full analysis validation failed after repair, using minimal fallback error=%s", exc2)
            return FullAnalysisSchema(
                fit_score=clamp_int(parsed.get("fit_score"), 0, 100, 0),
                criteria=[CriterionResultSchema(**c) for c in repaired_criteria],
            )


def _valid_source_lookup(sources_manifest: str) -> set[str]:
    valid = set()
    for line in sources_manifest.splitlines():
        parts = line.split("|")
        if parts:
            name = parts[0].strip()
            if name:
                valid.add(name)
    return valid


def _sanitize_source(value: str, valid_sources: set[str]) -> str:
    cleaned = (value or "").strip()
    return cleaned if cleaned in valid_sources else ""


_NEGATIVE_EVIDENCE_PATTERN = re.compile(
    r"^(no |none|not mentioned|not found|to confirm|no direct mention|no clear|"
    r"no signal|no specific|no explicit|unclear|no supporting)", re.IGNORECASE
)


def _is_positive_evidence(evidence: str) -> bool:
    text = (evidence or "").strip()
    return bool(text) and not _NEGATIVE_EVIDENCE_PATTERN.match(text)


def _ensure_single_highlight(text: str, phrase_source: str) -> str:
    if not text or not text.strip():
        return text
    if "**" in text:
        return text
    words = re.findall(r"[A-Za-z][A-Za-z\-']*", phrase_source or "")
    if len(words) < 2:
        return text
    phrase = " ".join(words[:3]) if len(words) >= 3 else " ".join(words[:2])
    lowered_text = text.lower()
    lowered_phrase = phrase.lower()
    position = lowered_text.find(lowered_phrase)
    if position == -1:
        return text
    return text[:position] + "**" + text[position:position + len(phrase)] + "**" + text[position + len(phrase):]


def _existing_names(entries: list[dict]) -> set[str]:
    return {(entry.get("name") or "").strip().lower() for entry in entries if entry.get("name")}


_NARRATIVE_PARTNERSHIP_PATTERN = re.compile(
    r"\b(?:partnership|partnered|partners?)\s+with\s+"
    r"((?:[A-Z][A-Za-z0-9&.\-]*)(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,4})"
)
_COMMON_WORD_STOPLIST = {
    "the", "government", "governments", "ngos", "ngo", "local", "state", "central",
    "schools", "communities", "partners", "various", "several", "multiple", "india",
}


def _extract_probable_partner_from_narrative(result: dict, company: str) -> dict | None:
    narrative_fields = (
        result.get("fit_rationale", ""),
        result.get("alignment_rationale", ""),
        result.get("delivery_model_evidence", ""),
        result.get("csr_head_note", ""),
    )
    for text in narrative_fields:
        if not text:
            continue
        clean_text = text.replace("**", "")
        for match in _NARRATIVE_PARTNERSHIP_PATTERN.finditer(clean_text):
            candidate = match.group(1).strip().rstrip(".,;: ")
            if not candidate or len(candidate) < 2:
                continue
            if candidate.strip().lower() in _COMMON_WORD_STOPLIST:
                continue
            if candidate.strip().lower() == company.strip().lower():
                continue
            if not re.match(r"^[A-Z]", candidate):
                continue
            return {
                "name": candidate,
                "relationship_type": "unclear",
                "source_excerpt": (
                    f"Named in the analysis narrative: \u201c...{match.group(0)}...\u201d — "
                    "not independently confirmed in the structured evidence extraction."
                ),
                "source": "",
                "confidence": "probable",
            }
    return None


def _reconcile_partners_with_narrative(result: dict, company: str) -> None:
    if result.get("partners"):
        return
    candidate = _extract_probable_partner_from_narrative(result, company)
    if candidate:
        logger.info(
            "partner reconciliation backfill added narrative-derived probable partner "
            "company=%r partner=%r — LLM named this in prose but omitted it from the "
            "structured array",
            company, candidate["name"],
        )
        result["partners"] = [candidate]
        open_questions = result.get("open_questions") or []
        note = f"Confirm the nature of {company}'s relationship with {candidate['name']} (funder, implementer, or co-designer) — surfaced from narrative text, not independently structured in the source evidence."
        if note not in open_questions:
            open_questions.append(note)
        result["open_questions"] = open_questions[:5]


def _reconcile_programmes_with_narrative(result: dict, company: str) -> None:
    if result.get("programmes"):
        return
    for partner in result.get("partners", []):
        name = (partner.get("name") or "").strip()
        excerpt = (partner.get("source_excerpt") or "")
        if not name:
            continue
        if "narrative" not in excerpt.lower():
            continue


def _backfill_narrative_gaps(result: dict, company: str, found_source_count: int = 0, sources: list | None = None) -> dict:
    _reconcile_partners_with_narrative(result, company)
    _reconcile_programmes_with_narrative(result, company)
    criteria = result.get("criteria", [])
    usable = [c for c in criteria if c.get("confidence", 0) > 0 and _is_positive_evidence(c.get("evidence", ""))]
    strongest = sorted(usable, key=lambda c: c.get("score", 0), reverse=True)
    weakest = sorted(
        [c for c in criteria if c.get("confidence", 0) > 0],
        key=lambda c: c.get("score", 0),
    )

    if not result.get("fit_rationale", "").strip() and strongest:
        top = strongest[0]["name"]
        bottom = weakest[0]["name"] if weakest else ""
        pieces = [f"{company}'s strongest signal is {top.lower()}"]
        if bottom and bottom != top:
            pieces.append(f"the weakest is {bottom.lower()}")
        result["fit_rationale"] = _ensure_single_highlight(
            f"Based on the available evidence, {'; '.join(pieces)}. "
            "This reflects the balance of confirmed signals rather than a single deciding factor.",
            top,
        )

    _backfill_contact_pathway_from_decision_makers(result)

    if sources:
        _backfill_linkedin_urls_from_people_search(result, sources)

    if not result.get("alignment_rationale", "").strip():
        alignment_candidates = [
            c for c in usable if c.get("id") in {"education_intervention", "stem", "tech_21cs", "public_schooling"}
        ]
        alignment_candidates.sort(key=lambda c: (c.get("score", 0), c.get("confidence", 0)), reverse=True)
        if alignment_candidates:
            best = alignment_candidates[0]
            result["alignment_rationale"] = _ensure_single_highlight(
                f"{company}'s disclosed activity shows {best.get('evidence')}, "
                f"which is the basis for the alignment estimate.",
                best.get("evidence", ""),
            )

    if not result.get("delivery_model_evidence", "").strip() and result.get("delivery_model") != "UNCLEAR":
        programmes = [p for p in result.get("programmes", []) if p.get("name")]
        if programmes:
            result["delivery_model_evidence"] = _ensure_single_highlight(
                f"Inferred from named activity such as {programmes[0].get('name')}.",
                programmes[0].get("name", ""),
            )

    sector = result.get("sector", {}) or {}
    if not sector.get("sector") or sector.get("sector") == "UNKNOWN":
        tech = next((c for c in usable if c.get("id") == "tech_21cs"), None)
        if tech:
            sector["sector"] = "Technology / Telecom"
            sector["reasoning"] = f"Inferred from disclosed activity: {tech.get('evidence')}."
            result["sector"] = sector

    if not result.get("source_quality_assessment", "").strip():
        if found_source_count > 0:
            result["source_quality_assessment"] = _ensure_single_highlight(
                "Findings draw on the sources actually fetched for this company; figures and "
                "claims should be **checked against source** before use.",
                "checked against source",
            )
        else:
            result["source_quality_assessment"] = (
                "No usable public sources were fetched for this company, so **no verified evidence** "
                "underlies this analysis."
            )

    if not result.get("evidence_recency", "").strip():
        result["evidence_recency"] = _ensure_single_highlight(
            "The fetched sources do not consistently state a publication date; treat recency as "
            "an unconfirmed detail until checked against the original page.",
            "unconfirmed detail",
        )

    if not result.get("csr_head_note", "").strip():
        decision_makers = [d for d in result.get("decision_makers", []) if d.get("name")]
        if decision_makers:
            result["csr_head_note"] = _ensure_single_highlight(
                f"{decision_makers[0]['name']} appears as the most public-facing contact in the sourced material.",
                "public-facing contact",
            )

    if result.get("overall_semantic_alignment", 0) == 0 and usable:
        alignment_ids = {"education_intervention", "stem", "tech_21cs", "public_schooling", "systems_change"}
        relevant = [c for c in usable if c.get("id") in alignment_ids] or usable
        result["overall_semantic_alignment"] = clamp_int(
            round((sum(c["score"] for c in relevant) / len(relevant)) / 5 * 100), 0, 100, 0
        )

    if result.get("overall_authenticity_score", 0) == 0 and found_source_count > 0:
        result["overall_authenticity_score"] = 40

    if result.get("fit_score", 0) == 0 and found_source_count > 0 and usable:
        alignment_ids = {"education_intervention", "stem", "tech_21cs", "public_schooling", "systems_change"}
        relevant = [c for c in usable if c.get("id") in alignment_ids] or usable
        inferred = clamp_int(
            round((sum(c["score"] for c in relevant) / len(relevant)) / 5 * 100), 0, 100, 0
        )
        if inferred > 0:
            result["fit_score"] = inferred

    return result


def _backfill_contact_pathway_from_decision_makers(result: dict) -> None:
    contact_pathway = result.get("contact_pathway") or {}
    if (contact_pathway.get("channel") or "").strip():
        return
    decision_makers = [d for d in result.get("decision_makers", []) if (d.get("name") or "").strip()]
    if not decision_makers:
        return
    decision_makers.sort(key=lambda d: d.get("public_facing_score", 0), reverse=True)
    top = decision_makers[0]
    title = (top.get("title") or "").strip()
    name = top["name"].strip()
    channel_text = (
        f"No open call was found; the warmest path is likely a direct approach to {name}"
        f"{f' ({title})' if title else ''} via {('their' if title else 'the')} CSR office."
    )
    contact_pathway["channel"] = _ensure_single_highlight(channel_text, name)
    result["contact_pathway"] = contact_pathway


def _backfill_linkedin_urls_from_people_search(result: dict, sources: list) -> None:
    people_source = next((s for s in sources if s.get("source_name") == "people_search"), None)
    hits = (people_source or {}).get("people_hits", [])
    if not hits:
        return
    hits_by_name = {}
    for hit in hits:
        name_key = (hit.get("name") or "").strip().lower()
        url = (hit.get("url") or "").strip()
        if name_key and url and name_key not in hits_by_name:
            hits_by_name[name_key] = url
    for decision_maker in result.get("decision_makers", []):
        if decision_maker.get("linkedin_url"):
            continue
        name_key = (decision_maker.get("name") or "").strip().lower()
        matched_url = hits_by_name.get(name_key)
        if not matched_url:
            for candidate_name, candidate_url in hits_by_name.items():
                if name_key and (name_key in candidate_name or candidate_name in name_key):
                    matched_url = candidate_url
                    break
        if matched_url:
            decision_maker["linkedin_url"] = _sanitize_linkedin_url(matched_url)


async def analyze_company(company: str, mission: str, sources: list, sources_manifest: str, temperature: float = 0.0) -> dict | None:
    structured_sources = structure_all_sources(sources, company)
    if not structured_sources:
        logger.info("analyze_company skipped company=%r reason=no_relevant_evidence", company)
        return None

    input_budget = analysis_input_token_budget(company, mission, sources_manifest)
    evidence_text = build_token_budgeted_evidence(structured_sources, company, input_budget)
    if not evidence_text.strip():
        logger.info("analyze_company skipped company=%r reason=empty_after_budgeting", company)
        return None

    prompt = full_company_analysis_prompt(company, mission, evidence_text, sources_manifest)
    prompt_tokens = estimate_tokens(prompt)
    output_ceiling = settings.anthropic_tpm_limit - OUTPUT_TOKEN_RESERVE

    shrink_attempts = 0
    while prompt_tokens > output_ceiling and input_budget > MIN_EVIDENCE_BUDGET and shrink_attempts < 6:
        overflow = prompt_tokens - output_ceiling
        input_budget = max(MIN_EVIDENCE_BUDGET, input_budget - overflow - 120)
        evidence_text = build_token_budgeted_evidence(structured_sources, company, input_budget)
        prompt = full_company_analysis_prompt(company, mission, evidence_text, sources_manifest)
        prompt_tokens = estimate_tokens(prompt)
        shrink_attempts += 1

    if shrink_attempts:
        logger.info(
            "analyze_company prompt shrunk company=%r attempts=%d final_input_budget=%d final_prompt_tokens=%d",
            company, shrink_attempts, input_budget, prompt_tokens,
        )

    if prompt_tokens > output_ceiling:
        logger.error(
            "analyze_company could not fit prompt within TPM budget company=%r prompt_tokens=%d ceiling=%d after %d shrink attempts anthropic_tpm_limit=%d",
            company, prompt_tokens, output_ceiling, shrink_attempts, settings.anthropic_tpm_limit,
        )
        return None

    raw_reply = await call_anthropic_chat(
        prompt,
        temperature=temperature,
        max_tokens=OUTPUT_TOKEN_RESERVE,
        caller=f"analyze_company:{company}",
    )

    if raw_reply is None:
        logger.error(
            "analyze_company got no reply company=%r — call_anthropic_chat failed (see prior log line)",
            company,
        )
        return None

    parsed = parse_json_response(raw_reply)
    validated = _repair_full_analysis(parsed)

    valid_sources = _valid_source_lookup(sources_manifest)
    result = validated.model_dump()

    by_id = {c["id"]: c for c in result["criteria"] if c["id"] in CRITERIA_IDS}
    ordered_criteria = []
    for criterion_id in CRITERIA_IDS:
        matched = by_id.get(criterion_id)
        score = clamp_float(matched.get("score") if matched else None, 0, 5, 0.0)
        confidence = clamp_int(matched.get("confidence") if matched else None, 0, 100, 0)
        evidence = (matched.get("evidence") if matched else "").strip() or "To confirm — no signal returned by model"
        reasoning = (matched.get("reasoning") if matched else "").strip()
        source = _sanitize_source(matched.get("source") if matched else "", valid_sources)
        ordered_criteria.append({
            "id": criterion_id,
            "name": CRITERIA_TITLES[criterion_id],
            "score": score,
            "confidence": confidence,
            "evidence": evidence[:240],
            "reasoning": reasoning[:240],
            "source": source,
        })
    result["criteria"] = ordered_criteria

    result["delivery_model_source"] = _sanitize_source(result.get("delivery_model_source", ""), valid_sources)
    result["spend"]["source"] = _sanitize_source(result["spend"].get("source", ""), valid_sources)
    result["spend"]["trend_source"] = _sanitize_source(result["spend"].get("trend_source", ""), valid_sources)
    for entry in result["spend"].get("history", []):
        entry["source"] = _sanitize_source(entry.get("source", ""), valid_sources)
    for programme in result["programmes"]:
        programme["source"] = _sanitize_source(programme.get("source", ""), valid_sources)
    for partner in result["partners"]:
        partner["source"] = _sanitize_source(partner.get("source", ""), valid_sources)
    for person in result["decision_makers"]:
        person["source"] = _sanitize_source(person.get("source", ""), valid_sources)
        person["linkedin_url"] = _sanitize_linkedin_url(person.get("linkedin_url", ""))
    for geography in result["geographies"]:
        geography["source"] = _sanitize_source(geography.get("source", ""), valid_sources)
    for flag in result["red_flags"]:
        flag["source"] = _sanitize_source(flag.get("source", ""), valid_sources)
    result["contact_pathway"]["source"] = _sanitize_source(result["contact_pathway"].get("source", ""), valid_sources)
    result["rfp_signal"]["source"] = _sanitize_source(result["rfp_signal"].get("source", ""), valid_sources)
    result["board_affinity"]["source"] = _sanitize_source(result["board_affinity"].get("source", ""), valid_sources)
    result["volunteering"]["source"] = _sanitize_source(result["volunteering"].get("source", ""), valid_sources)
    result["group_foundation"]["source"] = _sanitize_source(result["group_foundation"].get("source", ""), valid_sources)
    result["eligibility"]["source"] = _sanitize_source(result["eligibility"].get("source", ""), valid_sources)

    result["fit_score"] = clamp_int(result.get("fit_score"), 0, 100, 0)
    result["overall_semantic_alignment"] = clamp_int(result.get("overall_semantic_alignment"), 0, 100, 0)
    result["overall_authenticity_score"] = clamp_int(result.get("overall_authenticity_score"), 0, 100, 0)
    result["open_questions"] = [q.strip()[:200] for q in result.get("open_questions", []) if q and q.strip()][:5]
    result["llm_fallback_used"] = not bool(parsed)

    found_source_count = sum(1 for s in sources if s.get("status") == "FOUND")
    logger.info(
        "analyze_company model output company=%r fit_score=%d spend_disclosed=%s partners=%d programmes=%d fallback_used=%s",
        company, result["fit_score"], result["spend"].get("has_disclosed_budget"),
        len(result["partners"]), len(result["programmes"]), result["llm_fallback_used"],
    )

    result = _backfill_narrative_gaps(result, company, found_source_count, sources=sources)

    for programme in result["programmes"]:
        if not programme.get("source_excerpt") and programme.get("description"):
            programme["source_excerpt"] = programme["description"]
    for partner in result["partners"]:
        if not partner.get("source_excerpt") and partner.get("relationship_type"):
            partner["source_excerpt"] = f"Named as a {partner['relationship_type']} partner in the analysis."
    if result["spend"].get("has_disclosed_budget") and not result["spend"].get("source_excerpt"):
        result["spend"]["source_excerpt"] = "Figure derived from the analysis narrative."

    logger.info(
        "analyze_company DONE company=%r final_fit_score=%d final_partners=%d final_programmes=%d final_spend_disclosed=%s estimated_computed=%s",
        company, result["fit_score"], len(result["partners"]), len(result["programmes"]),
        result["spend"].get("has_disclosed_budget"), result["spend"].get("estimated_is_computed"),
    )
    return result


def important_links_prompt(company: str, search_results_text: str) -> str:
    return f"""Pick the best primary-source links on {company} CSR from these results. {company} must be the primary subject.

RESULTS:
\"\"\"
{search_results_text[:3500]}
\"\"\"

Pick up to 8, priority: 1) official CSR page, 2) MCA/CSR-2 or regulator pages, 3) annual/sustainability report PDF, 4) open call/RFP/partner page, 5) third-party coverage of {company}'s CSR. Exclude homepages, social media, job boards, unrelated news, wrong-company results.

Return ONLY valid JSON, URLs exactly as given:
{{"links": [{{"label": "<short label>", "url": "<exact url>", "relevance": "<why, <15 words>"}}]}}"""


async def select_important_links(company: str, search_results: list[dict]) -> list[dict]:
    if not search_results:
        return []

    lines = []
    for item in search_results[:20]:
        title = (item.get("title") or "").strip()
        url = (item.get("href") or item.get("url") or "").strip()
        body = (item.get("body") or item.get("snippet") or "").strip()
        if not url:
            continue
        lines.append(f"{title} | {url} | {body[:140]}")
    search_results_text = "\n".join(lines)
    if not search_results_text.strip():
        return []

    raw_reply = await call_anthropic_chat(
        important_links_prompt(company, search_results_text),
        temperature=0.0,
        max_tokens=500,
        caller=f"select_important_links:{company}",
    )
    parsed = parse_json_response(raw_reply)
    try:
        validated = ImportantLinksSchema.model_validate(parsed)
    except ValidationError:
        return []

    valid_urls = {(item.get("href") or item.get("url") or "").strip() for item in search_results}
    out = []
    for link in validated.links:
        if link.url not in valid_urls:
            continue
        out.append({"label": link.label.strip()[:80], "url": link.url, "relevance": link.relevance.strip()[:140]})
    logger.info("select_important_links company=%r candidates=%d selected=%d", company, len(search_results), len(out))
    return out[:8]


def people_match_prompt(company: str, raw_hits_text: str) -> str:
    return f"""Identify which LinkedIn results are current CSR/sustainability/foundation decision-makers at {company}, using only the snippets below.

HITS:
\"\"\"
{raw_hits_text[:3500]}
\"\"\"

For each plausible person: name, title, is_current_csr_role, match_confidence 0-100. Only include linkedin_url if a literal linkedin.com/in/ url is in input. "former"/"ex-"/"until 20XX"/"alumni" language: is_current_csr_role false, confidence capped at 30. tenure_status from date language: NEW_UNDER_1YR/ESTABLISHED_1_3YR/ENTRENCHED_3YR_PLUS/UNKNOWN.

Return ONLY valid JSON:
{{"people": [{{"name": "<name>", "title": "<title>", "is_current_csr_role": <bool>, "match_confidence": <0-100>, "linkedin_url": "<url or empty>", "tenure_status": "<NEW_UNDER_1YR|ESTABLISHED_1_3YR|ENTRENCHED_3YR_PLUS|UNKNOWN>", "reasoning": "<short>"}}]}}"""


async def match_people_from_search(company: str, hits: list[dict]) -> list[dict]:
    if not hits:
        return []
    lines = []
    for hit in hits[:20]:
        title = (hit.get("title") or "").strip()
        url = (hit.get("url") or hit.get("href") or "").strip()
        snippet = (hit.get("snippet") or hit.get("body") or "").strip()
        if not title and not snippet:
            continue
        lines.append(f"{title} | {url} | {snippet[:180]}")
    raw_hits_text = "\n".join(lines)
    if not raw_hits_text.strip():
        return []

    raw_reply = await call_anthropic_chat(
        people_match_prompt(company, raw_hits_text),
        temperature=0.0,
        max_tokens=1000,
        caller=f"match_people_from_search:{company}",
    )
    parsed = parse_json_response(raw_reply)
    try:
        validated = PeopleMatchListSchema.model_validate(parsed)
    except ValidationError:
        return []

    out = []
    for person in validated.people:
        if not person.name.strip():
            continue
        out.append({
            "name": person.name.strip(),
            "title": person.title.strip(),
            "is_current_csr_role": person.is_current_csr_role,
            "match_confidence": clamp_int(person.match_confidence, 0, 100, 0),
            "linkedin_url": _sanitize_linkedin_url(person.linkedin_url),
            "tenure_status": person.tenure_status if person.tenure_status in
                {"NEW_UNDER_1YR", "ESTABLISHED_1_3YR", "ENTRENCHED_3YR_PLUS", "UNKNOWN"} else "UNKNOWN",
            "reasoning": person.reasoning.strip(),
        })
    out.sort(key=lambda p: p["match_confidence"], reverse=True)
    filtered = [p for p in out if p["is_current_csr_role"] and p["match_confidence"] >= 50][:10]
    logger.info("match_people_from_search company=%r hits_in=%d matched_out=%d", company, len(hits), len(filtered))
    return filtered


def eligibility_and_group_prompt(company: str, evidence_text: str) -> str:
    return f"""Judge two things about {company} from the evidence: Companies Act Section 135 CSR mandate plausibility, and whether CSR runs through a separate parent/group foundation.

EVIDENCE:
\"\"\"
{evidence_text[:4000]}
\"\"\"

Section 135 thresholds (any one triggers it): net worth INR 500cr+, turnover INR 1000cr+, or net profit INR 5cr+. These are business-scale thresholds used only to judge mandate applicability — never restate them as CSR spend.

plausibly_mandated: LIKELY if a figure plausibly clears a threshold; UNLIKELY if clearly smaller; else UNKNOWN. Also extract net_worth_turnover_inr_crore and net_profit_inr_crore as plain numbers (INR crore) whenever explicitly stated, null otherwise — these feed a separate, code-computed estimate and must never be written into any CSR spend field.
routed_through_group: true only if a separate parent/group foundation is explicitly named.

Return ONLY valid JSON:
{{"eligibility": {{"plausibly_mandated": "<LIKELY|UNLIKELY|UNKNOWN>", "reasoning": "<short>", "net_worth_turnover_signal": "<short>", "net_worth_turnover_inr_crore": <number or null>, "net_profit_inr_crore": <number or null>}}, "group_foundation": {{"routed_through_group": <bool>, "foundation_name": "<name or empty>", "explanation": "<short>"}}}}"""


async def check_csr_eligibility(company: str, evidence_text: str) -> dict:
    if not evidence_text.strip():
        return {
            "eligibility": {
                "plausibly_mandated": "UNKNOWN", "reasoning": "", "net_worth_turnover_signal": "",
                "net_worth_turnover_inr_crore": None, "net_profit_inr_crore": None,
            },
            "group_foundation": {"routed_through_group": False, "foundation_name": "", "explanation": ""},
        }
    raw_reply = await call_anthropic_chat(
        eligibility_and_group_prompt(company, evidence_text),
        temperature=0.0,
        max_tokens=ELIGIBILITY_MAX_TOKENS,
        caller=f"check_csr_eligibility:{company}",
    )
    parsed = parse_json_response(raw_reply)
    eligibility = parsed.get("eligibility") or {}
    group_foundation = parsed.get("group_foundation") or {}
    net_worth_turnover = eligibility.get("net_worth_turnover_inr_crore")
    net_profit = eligibility.get("net_profit_inr_crore")
    return {
        "eligibility": {
            "plausibly_mandated": eligibility.get("plausibly_mandated", "UNKNOWN") if eligibility.get("plausibly_mandated") in {"LIKELY", "UNLIKELY", "UNKNOWN"} else "UNKNOWN",
            "reasoning": (eligibility.get("reasoning") or "").strip()[:280],
            "net_worth_turnover_signal": (eligibility.get("net_worth_turnover_signal") or "").strip()[:200],
            "net_worth_turnover_inr_crore": net_worth_turnover if isinstance(net_worth_turnover, (int, float)) else None,
            "net_profit_inr_crore": net_profit if isinstance(net_profit, (int, float)) else None,
        },
        "group_foundation": {
            "routed_through_group": bool(group_foundation.get("routed_through_group", False)),
            "foundation_name": (group_foundation.get("foundation_name") or "").strip()[:120],
            "explanation": (group_foundation.get("explanation") or "").strip()[:240],
        },
    }


def merge_eligibility_into_analysis(result: dict, eligibility_result: dict) -> dict:
    if not eligibility_result:
        return result
    current_eligibility = result.get("eligibility") or {}
    incoming_eligibility = eligibility_result.get("eligibility") or {}

    if current_eligibility.get("plausibly_mandated", "UNKNOWN") == "UNKNOWN" and incoming_eligibility.get("plausibly_mandated") != "UNKNOWN":
        current_eligibility["plausibly_mandated"] = incoming_eligibility.get("plausibly_mandated", "UNKNOWN")
    if not current_eligibility.get("reasoning") and incoming_eligibility.get("reasoning"):
        current_eligibility["reasoning"] = incoming_eligibility["reasoning"]
    if not current_eligibility.get("net_worth_turnover_signal") and incoming_eligibility.get("net_worth_turnover_signal"):
        current_eligibility["net_worth_turnover_signal"] = incoming_eligibility["net_worth_turnover_signal"]
    if current_eligibility.get("net_worth_turnover_inr_crore") is None and incoming_eligibility.get("net_worth_turnover_inr_crore") is not None:
        current_eligibility["net_worth_turnover_inr_crore"] = incoming_eligibility["net_worth_turnover_inr_crore"]
    if current_eligibility.get("net_profit_inr_crore") is None and incoming_eligibility.get("net_profit_inr_crore") is not None:
        current_eligibility["net_profit_inr_crore"] = incoming_eligibility["net_profit_inr_crore"]
    result["eligibility"] = current_eligibility

    current_group = result.get("group_foundation") or {}
    incoming_group = eligibility_result.get("group_foundation") or {}
    if not current_group.get("routed_through_group") and incoming_group.get("routed_through_group"):
        current_group["routed_through_group"] = True
        current_group["foundation_name"] = incoming_group.get("foundation_name", "")
        current_group["explanation"] = incoming_group.get("explanation", "")
    result["group_foundation"] = current_group

    if not result["spend"].get("has_disclosed_budget"):
        estimate, basis = _compute_statutory_estimate(current_eligibility)
        if estimate is not None and result["spend"].get("estimated_min_inr_crore") is None:
            result["spend"]["estimated_min_inr_crore"] = estimate
            result["spend"]["estimated_basis"] = basis
            result["spend"]["estimated_is_computed"] = True

    return result


def _classify_question_category(question: str) -> str:
    lowered = (question or "").lower()
    for category, keywords in QUESTION_CATEGORY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return category
    return "csr_policy"


def _rank_questions_for_resolution(open_questions: list[str], analysis: dict) -> list[tuple]:
    ranked = []
    for question in open_questions:
        category = _classify_question_category(question)
        priority = 0
        if category == "education_programme":
            priority = 3
        elif category == "csr_budget" and not (analysis.get("spend") or {}).get("has_disclosed_budget"):
            priority = 2
        elif category == "decision_maker" and not analysis.get("decision_makers"):
            priority = 2
        elif category == "ngo_partner" and not analysis.get("partners"):
            priority = 1
        else:
            priority = 0
        ranked.append((priority, category, question))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def question_resolution_prompt(company: str, question: str, followup_evidence: str) -> str:
    return f"""A CSR analyst had this specific open question about {company}: "{question}"

New evidence was just gathered specifically to try to answer it:
\"\"\"
{followup_evidence[:2500]}
\"\"\"

Does this new evidence answer the question? Only say answered=true if the evidence contains a concrete, specific fact directly resolving the question — not a generic or sector-wide statement. If it answers, give a one-sentence answer citing only what's in the evidence, and a confidence 0-100. If the evidence still does not resolve it, set answered=false, answer to a short note on what's still missing, and confidence to 0.

Optionally, if the answer contains a structured fact that maps cleanly onto one of: education_programme_name, education_programme_description, csr_spend_display, csr_spend_fiscal_year, csr_spend_inr_crore, decision_maker_name, decision_maker_title, decision_maker_linkedin_url, ngo_partner_name, ngo_partner_relationship — include it in "updates" as key-value pairs. Otherwise leave updates empty.

Return ONLY valid JSON:
{{"answered": <bool>, "answer": "<one sentence>", "confidence": <0-100>, "updates": {{}}}}"""


async def resolve_single_question(company: str, question: str, category: str, followup_evidence: str) -> dict | None:
    if not followup_evidence.strip():
        return None
    raw_reply = await call_anthropic_chat(
        question_resolution_prompt(company, question, followup_evidence),
        temperature=0.0,
        max_tokens=QUESTION_RESOLUTION_MAX_TOKENS,
        caller=f"resolve_question:{company}:{category}",
    )
    parsed = parse_json_response(raw_reply)
    try:
        validated = QuestionResolutionSchema.model_validate(parsed)
    except ValidationError:
        return None
    if not validated.answered or validated.confidence < 40:
        return None
    return {
        "question": question,
        "category": category,
        "answer": validated.answer.strip(),
        "confidence": clamp_int(validated.confidence, 0, 100, 0),
        "updates": validated.updates or {},
    }


def apply_question_resolution_to_analysis(result: dict, resolution: dict) -> dict:
    updates = resolution.get("updates") or {}
    category = resolution.get("category", "")

    if category == "education_programme" and updates.get("education_programme_name"):
        already_named = {(p.get("name") or "").strip().lower() for p in result.get("programmes", [])}
        name = updates["education_programme_name"].strip()
        if name and name.lower() not in already_named:
            result.setdefault("programmes", []).append({
                "name": name,
                "description": updates.get("education_programme_description", resolution.get("answer", ""))[:220],
                "is_multi_year": False,
                "cohort_or_scale": "",
                "source_excerpt": resolution.get("answer", "")[:200],
                "source": "",
                "confidence": "probable",
            })

    if category == "csr_budget" and updates.get("csr_spend_display"):
        spend = result.get("spend") or {}
        if not spend.get("has_disclosed_budget"):
            spend["display"] = updates["csr_spend_display"]
            spend["fiscal_year"] = updates.get("csr_spend_fiscal_year", "")
            inr_crore = updates.get("csr_spend_inr_crore")
            spend["inr_crore"] = inr_crore if isinstance(inr_crore, (int, float)) else spend.get("inr_crore")
            spend["has_disclosed_budget"] = True
            spend["confidence"] = min(resolution.get("confidence", 50), 70)
            spend["source_excerpt"] = resolution.get("answer", "")[:200]
            result["spend"] = spend

    if category == "decision_maker" and updates.get("decision_maker_name"):
        already_named = {(d.get("name") or "").strip().lower() for d in result.get("decision_makers", [])}
        name = updates["decision_maker_name"].strip()
        if name and name.lower() not in already_named:
            result.setdefault("decision_makers", []).append({
                "name": name,
                "title": updates.get("decision_maker_title", ""),
                "public_facing_score": min(resolution.get("confidence", 40), 60),
                "tenure_status": "UNKNOWN",
                "tenure_evidence": "",
                "source_excerpt": resolution.get("answer", "")[:200],
                "source": "",
                "linkedin_url": _sanitize_linkedin_url(updates.get("decision_maker_linkedin_url", "")),
            })

    if category == "ngo_partner" and updates.get("ngo_partner_name"):
        already_named = {(p.get("name") or "").strip().lower() for p in result.get("partners", [])}
        name = updates["ngo_partner_name"].strip()
        if name and name.lower() not in already_named:
            result.setdefault("partners", []).append({
                "name": name,
                "relationship_type": updates.get("ngo_partner_relationship", "unclear"),
                "source_excerpt": resolution.get("answer", "")[:200],
                "source": "",
                "confidence": "probable",
            })

    open_questions = result.get("open_questions") or []
    remaining = [q for q in open_questions if q != resolution.get("question")]
    verified_note = f"Verified via follow-up search: {resolution.get('answer', '')}"[:200]
    result["open_questions"] = ([verified_note] + remaining)[:5]
    result.setdefault("resolved_questions", []).append({
        "question": resolution.get("question", ""),
        "answer": resolution.get("answer", ""),
        "confidence": resolution.get("confidence", 0),
    })
    return result


async def resolve_open_questions(company: str, result: dict, search_module, search_cfg: dict,
                                  quota_guard=None, registry=None,
                                  max_questions: int = MAX_OPEN_QUESTIONS_TO_RESOLVE) -> dict:
    open_questions = result.get("open_questions") or []
    if not open_questions:
        return result

    ranked = _rank_questions_for_resolution(open_questions, result)
    to_resolve = [item for item in ranked if item[0] > 0][:max_questions]
    if not to_resolve:
        return result

    result.setdefault("resolved_questions", [])
    any_resolved = False

    for _, category, question in to_resolve:
        if anthropic_cooldown_remaining_seconds() > 0:
            logger.info("resolve_open_questions stopping early company=%r reason=cooldown_active", company)
            break
        try:
            followup_source = await search_module.run_targeted_queries(
                company, category, search_cfg, quota_guard=quota_guard, registry=registry,
            )
        except Exception as exc:
            logger.warning("resolve_open_questions followup search failed company=%r category=%r error=%s", company, category, exc)
            continue

        if followup_source.get("status") != "FOUND":
            logger.info("resolve_open_questions no new evidence company=%r category=%r question=%r", company, category, question)
            continue

        resolution = await resolve_single_question(company, question, category, followup_source.get("text", ""))
        if resolution is None:
            logger.info("resolve_open_questions unresolved company=%r category=%r question=%r", company, category, question)
            continue

        logger.info(
            "resolve_open_questions RESOLVED company=%r category=%r confidence=%d answer=%r",
            company, category, resolution["confidence"], resolution["answer"][:120],
        )
        result = apply_question_resolution_to_analysis(result, resolution)
        any_resolved = True

    result["open_questions_resolution_attempted"] = True
    result["open_questions_resolution_found_new_evidence"] = any_resolved
    return result


def strategic_insight_prompt(company: str, mission: str, state: str, fit_score: int, tier_label: str, analysis: dict) -> str:
    criteria_lines = "\n".join(
        f"- {c['name']}: {c['score']}/5 (conf {c['confidence']}%) — {c['evidence']}"
        for c in analysis.get("criteria", [])
    )
    red_flags = analysis.get("red_flags", []) or []
    red_flags_text = "; ".join(f"{r['flag']} ({r['severity']})" for r in red_flags) or "none"
    spend = analysis.get("spend", {}) or {}
    eligibility = analysis.get("eligibility", {}) or {}
    group_foundation = analysis.get("group_foundation", {}) or {}
    sector = analysis.get("sector", {}) or {}
    resolved_questions = analysis.get("resolved_questions") or []
    resolved_text = "; ".join(f"{r['question']} → {r['answer']}" for r in resolved_questions) or "none"
    estimate_note = ""
    if not spend.get("has_disclosed_budget") and spend.get("estimated_is_computed"):
        estimate_note = f"ESTIMATED (NOT DISCLOSED) STATUTORY MINIMUM: ₹{spend.get('estimated_min_inr_crore')} crore — {spend.get('estimated_basis', '')}"
    return f"""Senior CSR partnerships analyst writing the lead narrative of a due-diligence brief on {company} for education NGO TAP.

MISSION: {mission}
STATE: {state} · SCORE: {fit_score}/100 · TIER: {tier_label}
FIT RATIONALE FROM ANALYSIS: {analysis.get('fit_rationale', '')}
DELIVERY MODEL: {analysis.get('delivery_model', 'UNCLEAR')} — {analysis.get('delivery_model_evidence', '')}
ALIGNMENT: {analysis.get('overall_semantic_alignment', 0)}/100
CONTACT PATHWAY: {analysis.get('contact_pathway', {}).get('channel', '')}
SPEND TREND: {spend.get('trend_direction', 'UNKNOWN')} · CSR budget disclosed: {spend.get('has_disclosed_budget', False)}
{estimate_note}
SECTOR: {sector.get('sector', 'UNKNOWN')}
CSR-135 ELIGIBILITY: {eligibility.get('plausibly_mandated', 'UNKNOWN')}
GROUP FOUNDATION: {group_foundation.get('routed_through_group', False)} {('via ' + group_foundation.get('foundation_name', '')) if group_foundation.get('foundation_name') else ''}
FOLLOW-UP VERIFICATION RESULTS: {resolved_text}

SCORECARD:
{criteria_lines}

RED FLAGS: {red_flags_text}

Write one 180-300 word narrative in a measured, evidence-grounded tone — neither harsh nor inflated. Lead with genuine, evidence-backed strengths before caveats. State plainly whether/why this is a good fit based only on the analyst reasoning above; if spend.has_disclosed_budget is false, do not describe any revenue/turnover figure as CSR capacity — call it business scale only, and if an estimated statutory minimum is given above, you may cite it but must call it an estimate, never a disclosed figure. If follow-up verification results are present and not "none", weave in what was specifically checked and confirmed or ruled out — this is stronger evidence than the original pass and should be named as such. Name strongest/weakest dimensions without dwelling on the weakest; flag group-foundation routing and who to actually approach if relevant; note eligibility read if uncertain; give one concrete next step matching tier/model/pathway; flowing prose, not bullets. Do not treat unknown geography or unknown similarity to existing partners as a weakness. Treat any remaining open questions as items to verify next, not reasons the fit itself is weak.

{HIGHLIGHT_RULE.replace("(fit_rationale, alignment_rationale, delivery_model_evidence, source_quality_assessment, csr_head_note, evidence_recency, contact_pathway.channel, and every criterion evidence field)", "(this narrative)")}
Use exactly one bolded phrase somewhere in the narrative.

Return ONLY valid JSON:
{{"narrative": "<180-300 word narrative with one **2-3 word** highlight>"}}"""


async def generate_strategic_insight_narrative(company: str, mission: str, state: str, fit_score: int, tier_label: str, analysis: dict, temperature: float = 0.0) -> str:
    raw_reply = await call_anthropic_chat(
        strategic_insight_prompt(company, mission, state, fit_score, tier_label, analysis),
        temperature=temperature,
        max_tokens=INSIGHT_MAX_TOKENS,
        caller=f"strategic_insight:{company}",
    )
    parsed = parse_json_response(raw_reply)
    try:
        validated = StrategicInsightSchema.model_validate(parsed)
    except ValidationError:
        return LLM_UNAVAILABLE_EVIDENCE
    narrative = _normalize_highlight_markers(validated.narrative.strip())
    if narrative and "**" not in narrative:
        strongest = sorted(
            [c for c in analysis.get("criteria", []) if c.get("confidence", 0) > 0],
            key=lambda c: c.get("score", 0), reverse=True,
        )
        if strongest:
            narrative = _ensure_single_highlight(narrative, strongest[0].get("name", ""))
    return narrative if narrative else LLM_UNAVAILABLE_EVIDENCE


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