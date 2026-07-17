import functools
import json
import logging
import re
import time

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.config import settings
from app.pipeline.textproc import build_token_budgeted_evidence, estimate_tokens, structure_all_sources

logger = logging.getLogger("tap.llm")

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"

LLM_UNAVAILABLE_EVIDENCE = "LLM unavailable — unable to generate evidence"

OUTPUT_TOKEN_RESERVE = 5200
SCAFFOLD_SAFETY_MARGIN = 250
MIN_EVIDENCE_BUDGET = 350
INSIGHT_MAX_TOKENS = 500
ELIGIBILITY_MAX_TOKENS = 400
INTER_CALL_DELAY_SECONDS = 2.0

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


class ProgrammeSchema(BaseModel):
    name: str = ""
    description: str = Field(default="", max_length=220)
    is_multi_year: bool = False
    cohort_or_scale: str = ""
    source_excerpt: str = Field(default="", max_length=200)
    source: str = ""


class PartnerSchema(BaseModel):
    name: str = ""
    relationship_type: str = ""
    source_excerpt: str = Field(default="", max_length=200)
    source: str = ""


class DecisionMakerSchema(BaseModel):
    name: str = ""
    title: str = ""
    public_facing_score: int = Field(ge=0, le=100, default=0)
    tenure_status: str = "UNKNOWN"
    tenure_evidence: str = Field(default="", max_length=200)
    source_excerpt: str = Field(default="", max_length=200)
    source: str = ""


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
    "funding_capacity": "does the company's disclosed CSR budget look able to plausibly fund a grant of TAP's typical size",
    "csr_spend_trend": "rising multi-year=high, flat=medium, declining=low, no data=0",
    "decision_maker_tenure": "recently appointed=higher signal of new priorities, long entrenched/no signal=lower",
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
    "MARKER-HIGHLIGHT RULE (applies to fit_rationale, alignment_rationale, "
    "delivery_model_evidence, source_quality_assessment, csr_head_note, "
    "evidence_recency, contact_pathway.channel, and every criterion's evidence "
    "field): inside that field's text, wrap the single most decision-relevant "
    "phrase in **double asterisks**. The phrase must be exactly 2 to 3 words — "
    "never a full sentence, never a lone number, never more than 3 words. Choose "
    "the phrase a time-pressed fundraiser would want to see first, for example "
    "**routed through foundation**, **no open call**, **strong STEM focus**, "
    "**newly appointed head**, **spend is declining**. Each of these fields should "
    "carry exactly one bolded phrase if the field has any real content at all; "
    "leave a field with zero bolded phrases only if the field itself is genuinely "
    "empty. Never bold more than one phrase in the same field, and never bold "
    "anything in fields not listed here (name fields, titles, labels, source "
    "names, urls, booleans, enums)."
)

OUTPUT_ORDER_RULE = (
    "OUTPUT ORDER RULE: write the JSON object with fit_score, fit_rationale, "
    "overall_semantic_alignment, alignment_rationale, delivery_model, and "
    "delivery_model_evidence as the first six keys, in that order, before any "
    "other field. This protects the most important fields if you run low on "
    "output budget — everything after them (spend, programmes, criteria, etc.) "
    "is secondary detail and may be trimmed or abbreviated before the lead "
    "fields ever are."
)


def full_company_analysis_prompt(company: str, mission: str, evidence_text: str, sources_manifest: str) -> str:
    return f"""You are a thoughtful, generous, fair-minded CSR partnerships analyst judging whether {company} is a genuinely good funding/partnership fit for an Indian education NGO. Read the evidence below and form your own holistic judgment, giving the company every reasonable benefit of the doubt wherever the evidence is plausibly consistent with a good fit.

NGO MISSION: {mission}

EVIDENCE:
\"\"\"
{evidence_text}
\"\"\"

{OUTPUT_ORDER_RULE}

HARD RULE ON COMPLETENESS: every one of the 23 output fields below is equally mandatory — none is optional filler and none is secondary to the criteria scorecard. A blank narrative field (fit_rationale, alignment_rationale, delivery_model_evidence, source_quality_assessment, csr_head_note, evidence_recency) is only acceptable if the evidence text truly contains nothing usable for that field. If the evidence contains ANY relevant detail — even one sentence — you must write a real sentence for that field instead of leaving it blank or writing a placeholder like "Not provided" or "No supporting evidence found." Do not spend your effort on the criteria scorecard at the expense of the narrative fields; budget your output so every section gets at least one full sentence grounded in the evidence.

CRITICAL CONSISTENCY RULE: any specific fact you state in fit_rationale, alignment_rationale, csr_head_note, or any criterion's evidence MUST also be written into its matching structured field — the structured fields are the source of truth, and prose is only a summary of them, never a place to introduce facts that are absent elsewhere. Concretely: (a) if you name any programme, initiative, or campaign anywhere in your reasoning, it MUST have its own entry in the programmes array with that same name; (b) if you name any NGO, foundation, or organisation the company works with anywhere in your reasoning, it MUST have its own entry in the partners array; (c) if ANY evidence line states a revenue, turnover, net-worth, CSR-mandate percentage, or crore/lakh figure — even a computed or approximate one like "2% of ₹29,000 Cr" — you MUST set spend.has_disclosed_budget to true and put that figure in spend.display, even if it is an inferred minimum rather than a stated CSR line-item; (d) if you recommend a specific named person as the entry point anywhere in your reasoning, that same person's name MUST appear in contact_pathway.channel, AND that person MUST also appear in the decision_makers array with a title if one is known; (e) if you mention any state or city anywhere in your reasoning, it MUST have its own entry in the geographies array. Never mention a fact in prose while leaving its structured field empty. Before finalizing your JSON, re-read your own fit_rationale, alignment_rationale, csr_head_note and delivery_model_evidence one more time and check every named programme, organisation, person, place, and figure against the structured arrays — add any you missed.

{HIGHLIGHT_RULE}

1. FIT SCORE 0-100: your holistic judgment of how good a partnership fit this company is, considering the whole picture — do NOT compute this mechanically from the criteria below, and do NOT let a low criteria-scorecard average pull this number down. Sparse public sourcing is extremely common and is not itself evidence of poor fit — it usually just means less has been published, not that the company is a bad match. Score generously and directionally: any genuine positive signal (a named programme touching STEM, technology, 21st-century skills, or education; a credible co-design or platform-powering relationship; a plausible sector fit; scale and reputation consistent with an active CSR programme) should lift the score meaningfully rather than being offset by unrelated gaps like undisclosed budget or unconfirmed public-schooling focus — those are follow-up items for the analyst, not reasons to suppress the score. A large, well-known technology, IT-services, or professional-services company is close to the core of what CSR-funded education-technology NGOs look for as a prospect — treat that sector proximity itself as a meaningful positive, not a neutral fact. If the company's sector, scale, or business model plausibly aligns with education/technology/CSR-relevant work even without a fully documented programme, that alone justifies a mid-to-upper range score (60-75) rather than a low one, and if there is a named, concrete programme with real alignment to the mission (even one programme, even without financial disclosure), lean toward the 70-85 range. Reserve scores below 40 for cases where the evidence actively suggests poor fit (wrong sector, no CSR activity at all, explicit conflict with the mission) — thin, missing, or partially unresolved evidence on its own should not pull a plausible, well-aligned prospect down near zero; when in doubt between two adjacent bands, prefer the more generous one. A well-known, large, reputable company with a plausible sector fit but limited public CSR documentation should typically land in the 60-75 range, not near 0. Do not consider proximity to the NGO's operating states or overlap with the NGO's existing partners as scoring factors — mention them only as background color if relevant.
2. FIT RATIONALE (REQUIRED, 2-4 full sentences): explain the fit_score in plain language, referencing specific things found in the evidence (programmes, delivery model, decision-makers, spend, sector). Lead with what is genuinely promising before noting what remains to be confirmed. This must not be empty if fit_score is nonzero. Every programme, partner, or figure you reference here must also be present in its own structured field per the CRITICAL CONSISTENCY RULE above.
3. SEMANTIC ALIGNMENT 0-100 (REQUIRED, must not be 0 unless the company's sector/activity is genuinely unrelated to education, technology, or CSR entirely) + ALIGNMENT RATIONALE (REQUIRED, 1-2 full sentences grounded in evidence, or in the company's known sector/business if direct evidence is sparse) — how well the company's actual or plausible CSR activity overlaps with the NGO's mission area. A technology, professional-services, or education-adjacent company should score at least 50-60 here on sector plausibility alone, even with minimal direct evidence, and higher still if a named programme directly touches STEM, coding, or 21st-century skills.
4. DELIVERY MODEL: FUNDER/IMPLEMENTER/HYBRID/UNCLEAR + DELIVERY MODEL EVIDENCE (REQUIRED sentence naming the specific programme or statement that supports this classification — only use UNCLEAR with empty evidence if the text genuinely gives no clue).
5. BUDGET (do not leave blank if any number appears anywhere in the evidence): does any evidence disclose or imply an India CSR spend figure — including a stated revenue/turnover combined with a stated or standard CSR-mandate percentage, or even just a bare mandate percentage on its own (e.g. "mandatory 2% CSR spend")? Set has_disclosed_budget true and populate display with the figure (label it as an inferred minimum if computed, e.g. "~₹580 Cr (2% of ₹29,000 Cr revenue, inferred minimum)", or "~2% of net profit (exact figure not disclosed)" if only the percentage is known) whenever a real number is stated anywhere in the evidence, even approximate, derived, or partial. Only leave has_disclosed_budget false and display empty if the evidence contains no financial or mandate figures for this company at all. Give the latest figure if stated else null/conf 0; prior years into history[]; trend_direction RISING/FLAT/DECLINING/UNKNOWN from actual numbers only. An undisclosed budget is common for large companies and should be treated as an open question to verify, not as a negative signal in itself.
6. PROGRAMMES (do not leave empty if fit_rationale names any): list every named programme, initiative, campaign, or partnership mentioned anywhere in the evidence or referenced in your own fit_rationale, even briefly — multi-year vs one-off, scale if stated.
7. PARTNERS (do not leave empty if fit_rationale names any): list every NGO, foundation, or organisation the company is described as working with, funding, or partnering — including any named in your own fit_rationale, csr_head_note, or delivery_model_evidence — relationship_type funder/implementer/co-design/unclear.
8. DECISION MAKERS: every named leader, executive, or spokesperson quoted or mentioned in a CSR/sustainability context, title, public_facing_score 0-100, tenure_status. If you name this same person as the recommended contact in contact_pathway, they MUST appear here too.
9. GEOGRAPHY: every state/city mentioned anywhere in the evidence or in your own reasoning, for reference only.
10. RFP SIGNAL: explicit call for NGO partners — present, channel, evidence.
11. BOARD AFFINITY: named board/promoter personal education-philanthropy history.
12. VOLUNTEERING: named employee volunteering/payroll-giving touching education.
13. GROUP FOUNDATION: CSR run via separate parent/group foundation.
14. ELIGIBILITY: from net worth/turnover/profit figures, Section 135 applicability LIKELY/UNLIKELY/UNKNOWN.
15. SECTOR (REQUIRED — only UNKNOWN if the evidence gives no industry clue at all): classify using any company-description language in the evidence (e.g. telecom, IT services, manufacturing, FMCG, financial services), sub_sector if clear, with a one-line reasoning.
16. CRITERIA 0-5 each, all criteria ids in order, short evidence+reasoning, used only as supporting detail — not the basis for the fit score: {_criteria_rubric_block()}
17. RED FLAGS: genuine contradictions, marketing-not-substance signals, date mismatches. Severity low/medium/high. Do not invent flags just to lower the score, and do not list an unconfirmed detail (like an undisclosed budget or unstated public-schooling focus) as a red flag — those belong in open_questions instead.
18. CONTACT PATHWAY (REQUIRED — name the single most concrete real channel found, e.g. a specific named person with their title, a CSR page contact form, a foundation email; if you identify a recommended entry-point person anywhere in your reasoning, name them here AND add them to decision_makers with their title; only say "Not identified" if truly nothing exists).
19. EVIDENCE RECENCY (REQUIRED one full sentence): comment on how recent/current the evidence appears to be (years, fiscal years, or dated statements mentioned).
20. CSR HEAD NOTE (REQUIRED one full sentence): comment on leadership philosophy or approach based on any decision-maker quotes or statements found; if no decision-maker is named, comment instead on what the evidence suggests about how CSR is organised.
21. SOURCE QUALITY ASSESSMENT (REQUIRED 1-2 full sentences): assess how strong/weak/primary/secondary the sources actually used were. This field must always contain a real assessment — never leave it blank and never write a placeholder — even when only search snippets were available, say so plainly.
22. AUTHENTICITY SCORE 0-100 (REQUIRED, must not default to 0 unless sourcing is truly untrustworthy): how much you trust the sourcing, for reference only.
23. OPEN QUESTIONS: up to 5 short items to verify. Use this field, not the fit score or red flags, to carry unconfirmed details like undisclosed budget, unclear public-schooling focus, or unclear foundation routing.

All criteria ids below must appear exactly once, in order. Missing evidence for a criterion: score 0, confidence 0, evidence "To confirm — no signal in evidence".

Rules: evidence fields are paraphrases under 20 words, never verbatim. Never fabricate facts, but never leave a required narrative field blank when the evidence contains anything relevant — write the sentence. Numbers internally consistent. Keep every string concise so the full reply fits within {OUTPUT_TOKEN_RESERVE} output tokens, but prioritize filling every required field over verbosity in any single one, and prioritize the first six keys above all else per the OUTPUT ORDER RULE. Apply the marker-highlight rule above inside the relevant fields. Reply with ONE JSON object, nothing else.

JSON shape:
{{
  "fit_score": <int 0-100>,
  "fit_rationale": "<2-4 sentences, required, one **2-3 word** highlight>",
  "overall_semantic_alignment": <int 0-100>,
  "alignment_rationale": "<1-2 sentences, required, one **2-3 word** highlight>",
  "delivery_model": "<FUNDER|IMPLEMENTER|HYBRID|UNCLEAR>",
  "delivery_model_evidence": "<sentence, required unless truly no clue, one **2-3 word** highlight>",
  "spend": {{"inr_crore": <number or null>, "display": "<as stated>", "fiscal_year": "<if stated>", "has_disclosed_budget": <bool>, "confidence": <0-100>, "source_excerpt": "<short>", "trend_direction": "<RISING|FLAT|DECLINING|UNKNOWN>", "trend_evidence": "<short>", "history": [{{"fiscal_year": "<year>", "inr_crore": <number or null>, "display": "<as stated>", "source_excerpt": "<short>"}}]}},
  "programmes": [{{"name": "<name>", "description": "<short>", "is_multi_year": <bool>, "cohort_or_scale": "<if stated>", "source_excerpt": "<short>"}}],
  "partners": [{{"name": "<name>", "relationship_type": "<funder|implementer|co-design|unclear>", "source_excerpt": "<short>"}}],
  "decision_makers": [{{"name": "<name>", "title": "<title>", "public_facing_score": <0-100>, "tenure_status": "<NEW_UNDER_1YR|ESTABLISHED_1_3YR|ENTRENCHED_3YR_PLUS|UNKNOWN>", "tenure_evidence": "<short>", "source_excerpt": "<short>"}}],
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
  "eligibility": {{"plausibly_mandated": "<LIKELY|UNLIKELY|UNKNOWN>", "reasoning": "<short>", "net_worth_turnover_signal": "<short>"}},
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
    return max(MIN_EVIDENCE_BUDGET, budget)


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

    since_last_call = time.monotonic() - _last_call_finished_at_monotonic
    if since_last_call < INTER_CALL_DELAY_SECONDS:
        wait_seconds = INTER_CALL_DELAY_SECONDS - since_last_call
        logger.info("anthropic pacing delay caller=%s waiting=%.1fs", caller, wait_seconds)
        time.sleep(wait_seconds)

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
        "anthropic request caller=%s model=%s max_tokens=%d prompt_chars=%d estimated_prompt_tokens=%d window_used_before=%d",
        caller, resolved_model, max_tokens, len(prompt), estimated_prompt_tokens, tokens_used_in_window,
    )
    request_started_at = time.monotonic()
    _record_tpm_usage(estimated_total_tokens)

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
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
        return recovered
    end = max(cleaned.rfind("}"), cleaned.rfind("]"))
    if end == -1:
        return {}
    for start_offset in range(0, 3):
        try:
            parsed = json.loads(cleaned[: end + 1 - start_offset] + "}" * start_offset)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            continue
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


def _normalize_highlight_markers(text: str) -> str:
    if not text:
        return text
    cleaned = _STRAY_MARKER_PATTERN.sub("**", text)
    if len(_UNPAIRED_DOUBLE_STAR_PATTERN.findall(cleaned)) % 2 != 0:
        cleaned = cleaned.replace("**", "")
    return cleaned


def _repair_full_analysis(parsed: dict) -> FullAnalysisSchema:
    if not isinstance(parsed, dict):
        parsed = {}
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
    except ValidationError:
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
        except ValidationError:
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


_TITLE_CASE_PHRASE_PATTERN = re.compile(
    r"\b(?:[A-Z][a-zA-Z&()'\-]*\s+){0,4}[A-Z][a-zA-Z&()'\-]*"
    r"(?:\s+(?:Project|Programme|Program|Academy|Initiative|Mission|Foundation|"
    r"Trust|Fund|Fellowship|Scholarship|Chatbot|Campaign|Labs?))\b"
)
_GENERIC_PARTNER_STOPWORDS = {
    "csr", "india", "the company", "this company", "tap", "ngo", "ngos",
    "government", "govt", "delhi", "mumbai", "maharashtra",
}
_PARTNER_LIST_PATTERN = re.compile(
    r"\b(?:named implementing partners?|implementing partners?|partners? like|"
    r"partners? including|partners? such as|funds? implementing partners?"
    r"(?: to deliver[^(]*)?)\s*\(?\s*([A-Z][\w&.'\- ]{1,60}"
    r"(?:\s*,\s*[A-Z][\w&.'\- ]{1,60}){0,8}(?:\s+and\s+[A-Z][\w&.'\- ]{1,60})?)\)?",
    re.IGNORECASE,
)
_PARTNERSHIP_MENTION_PATTERN = re.compile(
    r"\b(?:through its|via its|its)\s+([A-Z][\w&.'\- ]{2,50}?)\s+partnership\b"
    r"|\bpartnership with\s+([A-Z][\w&.'\- ]{2,50})\b"
    r"|\b([A-Z][\w&.'\- ]{2,50}?)\s+(?:Foundation|Trust)\s+partnership\b",
    re.IGNORECASE,
)
_NGO_FOUNDATION_PATTERN = re.compile(
    r"\b([A-Z][a-zA-Z&.'\-]*(?:\s+[A-Z][a-zA-Z&.'\-]*){0,4}"
    r"\s+(?:Foundation|Trust|NGO|Society|Charitable Trust))\b"
)
_MONEY_PATTERN = re.compile(
    r"(₹\s?[\d,]+(?:\.\d+)?\s?(?:crore|cr\.?|lakh|lac)|"
    r"(?:INR|Rs\.?)\s?[\d,]+(?:\.\d+)?\s?(?:crore|cr\.?|lakh|lac)?|"
    r"[\d,]+(?:\.\d+)?\s?(?:crore|cr\.?)\b)",
    re.IGNORECASE,
)
_PERCENT_MANDATE_PATTERN = re.compile(r"\b(\d(?:\.\d+)?)\s?%\s?(?:csr)?\s?(?:mandate|of)", re.IGNORECASE)
_PERSON_NAME_PATTERN = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b"
    r"(?:,?\s*(?:VP|Vice President|Head|Director|CEO|Officer|Manager|Lead)[^.]{0,60})?"
)
_STATE_NAME_PATTERN = re.compile(
    r"\b(Delhi|Mumbai|Maharashtra|Karnataka|Bengaluru|Bangalore|Tamil Nadu|Chennai|"
    r"Telangana|Hyderabad|Gujarat|Ahmedabad|West Bengal|Kolkata|Uttar Pradesh|Lucknow|"
    r"Rajasthan|Jaipur|Punjab|Haryana|Kerala|Odisha|Bihar|Madhya Pradesh|Pune|"
    r"Andhra Pradesh|Assam|Goa|Chandigarh)\b"
)


def _extract_named_entities(text: str, pattern: re.Pattern, limit: int) -> list[str]:
    seen, out = set(), []
    for match in pattern.finditer(text or ""):
        candidate = (match.group(1) if match.groups() else match.group(0)).strip()
        key = candidate.lower()
        if len(candidate) < 4 or key in seen:
            continue
        seen.add(key)
        out.append(candidate)
        if len(out) >= limit:
            break
    return out


def _extract_partner_list_names(text: str, limit: int) -> list[str]:
    seen, out = set(), []
    for match in _PARTNER_LIST_PATTERN.finditer(text or ""):
        chunk = match.group(1)
        chunk = re.sub(r"\s+and\s+", ", ", chunk)
        for raw_name in chunk.split(","):
            candidate = raw_name.strip(" .()")
            key = candidate.lower()
            if len(candidate) < 2 or key in _GENERIC_PARTNER_STOPWORDS or key in seen:
                continue
            seen.add(key)
            out.append(candidate)
            if len(out) >= limit:
                return out
    for match in _PARTNERSHIP_MENTION_PATTERN.finditer(text or ""):
        for group in match.groups():
            if not group:
                continue
            candidate = group.strip(" .()")
            key = candidate.lower()
            if len(candidate) < 2 or key in _GENERIC_PARTNER_STOPWORDS or key in seen:
                continue
            seen.add(key)
            out.append(candidate)
            if len(out) >= limit:
                return out
    return out


def _existing_names(entries: list[dict]) -> set[str]:
    return {(entry.get("name") or "").strip().lower() for entry in entries if entry.get("name")}


def _backfill_from_rationale(result: dict, rationale_text: str) -> None:
    if not rationale_text or not rationale_text.strip():
        return

    existing_programme_names = _existing_names(result.get("programmes", []))
    for name in _extract_named_entities(rationale_text, _TITLE_CASE_PHRASE_PATTERN, 8):
        if name.lower() in existing_programme_names:
            continue
        existing_programme_names.add(name.lower())
        result.setdefault("programmes", []).append({
            "name": name,
            "description": "Referenced in the analysis narrative as a named initiative.",
            "is_multi_year": "multi-year" in rationale_text.lower() or "sustained" in rationale_text.lower(),
            "cohort_or_scale": "",
            "source_excerpt": "",
            "source": "",
        })

    existing_partner_names = _existing_names(result.get("partners", []))
    partner_candidates = (
        _extract_partner_list_names(rationale_text, 8)
        + _extract_named_entities(rationale_text, _NGO_FOUNDATION_PATTERN, 6)
    )
    for name in partner_candidates:
        if name.lower() in existing_partner_names:
            continue
        existing_partner_names.add(name.lower())
        if "co-design" in rationale_text.lower():
            relationship_type = "co-design"
        elif re.search(r"implement(ing|ation) partner", rationale_text, re.IGNORECASE):
            relationship_type = "implementer"
        elif re.search(r"\bfund(s|ed|ing)?\b", rationale_text, re.IGNORECASE):
            relationship_type = "funder"
        else:
            relationship_type = "unclear"
        result.setdefault("partners", []).append({
            "name": name,
            "relationship_type": relationship_type,
            "source_excerpt": "",
            "source": "",
        })

    spend = result.get("spend") or {}
    if not spend.get("has_disclosed_budget"):
        money_match = _MONEY_PATTERN.search(rationale_text)
        percent_match = _PERCENT_MANDATE_PATTERN.search(rationale_text)
        if money_match or percent_match:
            if money_match:
                display = money_match.group(0).strip()
                if percent_match:
                    display += f" ({percent_match.group(1)}% CSR mandate, inferred minimum)"
            else:
                display = f"~{percent_match.group(1)}% CSR mandate applies (exact spend not disclosed)"
            spend["has_disclosed_budget"] = True
            spend["display"] = display
            spend["confidence"] = spend.get("confidence") or 35
            spend["source_excerpt"] = "Derived from figures mentioned in the analysis narrative."
            result["spend"] = spend

    existing_geo_names = _existing_names(result.get("geographies", []))
    for place in _extract_named_entities(rationale_text, _STATE_NAME_PATTERN, 6):
        if place.lower() in existing_geo_names:
            continue
        existing_geo_names.add(place.lower())
        result.setdefault("geographies", []).append({
            "place": place,
            "source_excerpt": "Mentioned in the analysis narrative.",
            "source": "",
        })

    contact_pathway = result.get("contact_pathway") or {}
    if not (contact_pathway.get("channel") or "").strip():
        name_title_match = re.search(
            r"(?:via|through|contact|approach)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\s*\(([^)]{3,60})\)",
            rationale_text,
        )
        if name_title_match:
            contact_pathway["channel"] = (
                f"Warm outreach via {name_title_match.group(1)} ({name_title_match.group(2)}), "
                f"named in the analysis narrative as the recommended entry point."
            )
            result["contact_pathway"] = contact_pathway
        else:
            bare_name_match = re.search(
                r"(?:via|through|contact|approach)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b",
                rationale_text,
            )
            if bare_name_match:
                contact_pathway["channel"] = (
                    f"Warm outreach via {bare_name_match.group(1)}, "
                    f"named in the analysis narrative as the recommended entry point."
                )
                result["contact_pathway"] = contact_pathway


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


def _backfill_narrative_gaps(result: dict, company: str, found_source_count: int = 0) -> dict:
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
            "This score reflects the balance of those signals rather than a single deciding factor.",
            top,
        )

    for narrative_source in (
        result.get("fit_rationale", ""),
        result.get("csr_head_note", ""),
        result.get("alignment_rationale", ""),
        result.get("delivery_model_evidence", ""),
        result.get("source_quality_assessment", ""),
        *[c.get("evidence", "") for c in criteria],
    ):
        _backfill_from_rationale(result, narrative_source)

    _backfill_contact_pathway_from_decision_makers(result)

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
                "Findings draw primarily on company-published material and press coverage rather than "
                "independent third-party verification; figures and claims should be **checked against source** "
                "before use.",
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
        result["overall_authenticity_score"] = 55

    if result.get("fit_score", 0) == 0 and found_source_count > 0 and usable:
        alignment_ids = {"education_intervention", "stem", "tech_21cs", "public_schooling", "systems_change"}
        relevant = [c for c in usable if c.get("id") in alignment_ids] or usable
        inferred = clamp_int(
            round((sum(c["score"] for c in relevant) / len(relevant)) / 5 * 100), 0, 100, 0
        )
        if inferred > 0:
            result["fit_score"] = max(inferred, 45)

    return result


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
    result = _backfill_narrative_gaps(result, company, found_source_count)

    for programme in result["programmes"]:
        if not programme.get("source_excerpt") and programme.get("description"):
            programme["source_excerpt"] = programme["description"]
    for partner in result["partners"]:
        if not partner.get("source_excerpt") and partner.get("relationship_type"):
            partner["source_excerpt"] = f"Named as a {partner['relationship_type']} partner in the analysis."
    if result["spend"].get("has_disclosed_budget") and not result["spend"].get("source_excerpt"):
        result["spend"]["source_excerpt"] = "Figure derived from the analysis narrative."

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
            "linkedin_url": person.linkedin_url.strip(),
            "tenure_status": person.tenure_status if person.tenure_status in
                {"NEW_UNDER_1YR", "ESTABLISHED_1_3YR", "ENTRENCHED_3YR_PLUS", "UNKNOWN"} else "UNKNOWN",
            "reasoning": person.reasoning.strip(),
        })
    out.sort(key=lambda p: p["match_confidence"], reverse=True)
    return [p for p in out if p["is_current_csr_role"] and p["match_confidence"] >= 50][:10]


def eligibility_and_group_prompt(company: str, evidence_text: str) -> str:
    return f"""Judge two things about {company} from the evidence: Companies Act Section 135 CSR mandate plausibility, and whether CSR runs through a separate parent/group foundation.

EVIDENCE:
\"\"\"
{evidence_text[:4000]}
\"\"\"

Section 135 thresholds (any one triggers it): net worth INR 500cr+, turnover INR 1000cr+, or net profit INR 5cr+.

plausibly_mandated: LIKELY if a figure plausibly clears a threshold; UNLIKELY if clearly smaller; else UNKNOWN.
routed_through_group: true only if a separate parent/group foundation is explicitly named.

Return ONLY valid JSON:
{{"eligibility": {{"plausibly_mandated": "<LIKELY|UNLIKELY|UNKNOWN>", "reasoning": "<short>", "net_worth_turnover_signal": "<short>"}}, "group_foundation": {{"routed_through_group": <bool>, "foundation_name": "<name or empty>", "explanation": "<short>"}}}}"""


async def check_csr_eligibility(company: str, evidence_text: str) -> dict:
    if not evidence_text.strip():
        return {
            "eligibility": {"plausibly_mandated": "UNKNOWN", "reasoning": "", "net_worth_turnover_signal": ""},
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
    return {
        "eligibility": {
            "plausibly_mandated": eligibility.get("plausibly_mandated", "UNKNOWN") if eligibility.get("plausibly_mandated") in {"LIKELY", "UNLIKELY", "UNKNOWN"} else "UNKNOWN",
            "reasoning": (eligibility.get("reasoning") or "").strip()[:280],
            "net_worth_turnover_signal": (eligibility.get("net_worth_turnover_signal") or "").strip()[:200],
        },
        "group_foundation": {
            "routed_through_group": bool(group_foundation.get("routed_through_group", False)),
            "foundation_name": (group_foundation.get("foundation_name") or "").strip()[:120],
            "explanation": (group_foundation.get("explanation") or "").strip()[:240],
        },
    }


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
    return f"""Senior CSR partnerships analyst writing the lead narrative of a due-diligence brief on {company} for education NGO TAP.

MISSION: {mission}
STATE: {state} · SCORE: {fit_score}/100 · TIER: {tier_label}
FIT RATIONALE FROM ANALYSIS: {analysis.get('fit_rationale', '')}
DELIVERY MODEL: {analysis.get('delivery_model', 'UNCLEAR')} — {analysis.get('delivery_model_evidence', '')}
ALIGNMENT: {analysis.get('overall_semantic_alignment', 0)}/100
CONTACT PATHWAY: {analysis.get('contact_pathway', {}).get('channel', '')}
SPEND TREND: {spend.get('trend_direction', 'UNKNOWN')} · budget disclosed: {spend.get('has_disclosed_budget', False)}
SECTOR: {sector.get('sector', 'UNKNOWN')}
CSR-135 ELIGIBILITY: {eligibility.get('plausibly_mandated', 'UNKNOWN')}
GROUP FOUNDATION: {group_foundation.get('routed_through_group', False)} {('via ' + group_foundation.get('foundation_name', '')) if group_foundation.get('foundation_name') else ''}

SCORECARD:
{criteria_lines}

RED FLAGS: {red_flags_text}

Write one 180-300 word narrative in a fair, encouraging tone — not harsh or dismissive, and not inflating either. Lead with genuine strengths before caveats. States plainly whether/why this is a good fit grounded in the analyst's own reasoning above; names strongest/weakest dimensions without dwelling on the weakest; flags group-foundation routing and who to actually approach if relevant; notes eligibility read if uncertain; gives one concrete next step matching tier/model/pathway; flowing prose, not bullets. Do not treat unknown geography or unknown similarity to existing partners as a weakness. Treat open questions (undisclosed budget, unconfirmed public-schooling focus, etc.) as items to verify next, not as reasons the fit itself is weak.

{HIGHLIGHT_RULE.replace("(applies to fit_rationale, alignment_rationale, delivery_model_evidence, source_quality_assessment, csr_head_note, evidence_recency, contact_pathway.channel, and every criterion's evidence field)", "(applies to this narrative)")}
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