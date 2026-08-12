import json
import logging
import re
import typing

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.config import settings
from app.pipeline.textproc import combine_evidence_text, estimate_tokens

logger = logging.getLogger("tap.llm")

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"

LLM_UNAVAILABLE_EVIDENCE = "LLM unavailable — unable to generate evidence"
LLM_SCORING_UNAVAILABLE_NOTE = (
    "Automated scoring could not complete for this run, but the facts below were "
    "successfully extracted from the fetched sources — verify and score manually."
)

OUTPUT_TOKEN_RESERVE = 6000
EXTRACTION_OUTPUT_TOKEN_RESERVE = 4200
MIN_EVIDENCE_TOKEN_BUDGET = 500
ANTHROPIC_REQUEST_TIMEOUT_SECONDS = 120.0
MIN_PROMPT_TRIM_CHARS = 150
MAX_PROMPT_SHRINK_ATTEMPTS = 6
PROMPT_SHRINK_SAFETY_MARGIN = 120
DEFAULT_ANTHROPIC_CONTEXT_WINDOW = 200000

EXTRACTION_PRIORITY_KEYS = [
    "overall_authenticity_score",
    "source_quality_assessment",
    "evidence_recency",
    "delivery_model",
    "delivery_model_evidence",
    "sector",
    "eligibility",
    "spend",
]


def _anthropic_context_window() -> int:
    value = getattr(settings, "anthropic_context_window", None)
    if not isinstance(value, int) or value <= 0:
        logger.warning(
            "settings.anthropic_context_window missing or invalid (%r) — falling back to default=%d. "
            "Add anthropic_context_window to app.config.Settings to remove this warning.",
            value, DEFAULT_ANTHROPIC_CONTEXT_WINDOW,
        )
        return DEFAULT_ANTHROPIC_CONTEXT_WINDOW
    return value

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

# NOTE ON RUBRIC WORDING: each rubric line is deliberately anchored to an observable
# fact pattern (named programme, named person, stated figure) rather than to a vague
# impression, precisely so that two independent runs over the SAME evidence converge
# on the same score. Vague, impression-based rubric lines are the main way scoring
# drifts between runs even at temperature=0 — the model has more freedom to "vibe"
# a score when the rubric doesn't pin it to a concrete artifact in the evidence.
_RUBRIC = {
    "education_intervention": "hands-on programme, not a scholarship or one-off donation",
    "stem": "named STEM/coding/robotics/science exposure",
    "tech_21cs": "tech-delivered learning or 21st-century-skills content",
    "public_schooling": "explicit government-school work; absence alone doesn't disqualify",
    "systems_change": "teacher training, measured outcomes, scale, or policy influence",
    "programme_depth": "one-off activity scores lower; named multi-year programme scores higher",
    "partnership_quality": "named, multi-year NGO partner scores higher; give real credit if the company already funds other education/skilling-adjacent NGOs, even ones unrelated to TAP",
    "decision_maker_accessibility": "a named individual whose title or evidence context is specifically CSR/education/foundation-related, not merely any employee — see decision-maker exclusion rule",
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

# CONSISTENCY RULE (feedback item 1): the single largest driver of run-to-run score
# drift is the model picking a "vibe" score instead of deriving it mechanically from
# the rubric anchors below. This block forces every score to cite the specific fact
# pattern behind it, which is both more auditable and more repeatable across runs
# that see the same evidence.
SCORING_CONSISTENCY_RULE = (
    "CONSISTENCY (read before scoring): scores must be repeatable — if this exact evidence "
    "were scored again, you should land on the same numbers. To guarantee that, never assign "
    "a criterion score from general impression, company reputation, or brand size. For every "
    "criterion, first identify the single most relevant fact/quote in the extracted evidence, "
    "then map that fact to a score using the rubric line for that criterion, and write that "
    "fact (not a vibe) into the `evidence` field. If two criteria could plausibly take the same "
    "score for the same underlying reason, they should — do not vary scores across similar "
    "criteria without a distinct evidentiary reason for each. Do not let overall enthusiasm "
    "about the company inflate individual criterion scores beyond what each one's own evidence "
    "supports."
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

# EDUCATION SPEND RULE (feedback item 2): fundraisers said the overall CSR total is not
# useful on its own — they need the education-specific slice AND a 2-3 year trend AND an
# explicit up/down/flat read. The additions below make trend-hunting mandatory rather
# than opportunistic, and make total-only figures clearly secondary in the output.
EDUCATION_SPEND_RULE = (
    "EDUCATION SPEND VS TOTAL CSR (second common error): most companies run CSR across several "
    "causes, not just education. spend.inr_crore / spend.display / spend.fiscal_year / "
    "spend.trend_* must hold ONLY the education-specific slice — set is_education_specific=true "
    "and populate these only when the evidence states an education-specific figure or "
    "percentage. If the evidence only gives a TOTAL CSR figure with no education breakdown, "
    "leave spend.is_education_specific=false and spend.inr_crore/display/fiscal_year empty, and "
    "put that total in total_csr_inr_crore / total_csr_display / total_csr_fiscal_year instead, "
    "clearly labeled as total CSR, never as the education budget. "
    "TREND IS MANDATORY WHENEVER POSSIBLE: actively scan the evidence for figures from more "
    "than one fiscal year — annual/CSR reports often disclose 2-3 years side by side even when "
    "only the latest year is prominently mentioned. Populate spend.history[] with every "
    "distinct (fiscal_year, figure) pair you find, education-specific if available, else total "
    "CSR clearly marked as such via is_education_specific=false entries. Set trend_direction to "
    "RISING/FLAT/DECLINING whenever two or more years of figures exist anywhere in the evidence "
    "— UNKNOWN should only be used when truly one data point, or none, is available. A single "
    "year's figure with no trend is materially less useful to a fundraiser than a 2-3 year "
    "trend, so treat finding the trend as equally important as finding the headline number."
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
    "leaving it out of the structured list. "
    "PARTNER HISTORY DEPTH (feedback priority): a partner list is far more valuable to a "
    "fundraiser when it spans multiple years, because it reveals whether the relationship is "
    "one-off or sustained, and what scale of grant the company typically gives. Actively check "
    "the evidence for partner mentions across different fiscal years / annual reports, not just "
    "the most recent one, and include each distinct (partner, year) combination you find as a "
    "separate entry rather than collapsing multi-year partners into a single undated entry — "
    "this lets the reader see the funding pattern over time."
)

# PROGRAMME RULE (feedback items 3 + 6): the #1 complaint was that focus-area labels like
# "financial literacy" are useless without knowing exactly what's funded, who benefits, and
# through what delivery channel — because that determines real overlap with TAP's model.
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
    "programmes array. "
    "GO BEYOND THE THEME LABEL: a theme word ('financial literacy', 'skill development', "
    "'life skills') is not on its own useful for a fundraising decision, because the same theme "
    "can mean completely different things in practice — e.g. banking-awareness workshops for "
    "unemployed adults versus a curriculum-embedded financial-literacy module for school "
    "children are both 'financial literacy' but have opposite relevance to TAP. For every "
    "programme, the `description` field must therefore state, wherever the evidence allows: "
    "(a) the delivery channel — in-school/curriculum-embedded, standalone adult workshop, "
    "digital/app-based, vocational/on-the-job, or other, (b) the concrete beneficiary (school-"
    "going children vs out-of-school youth vs adults vs teachers), and (c) whether it is "
    "one-off or ongoing. If the evidence truly does not support any of (a)-(c) beyond the theme "
    "name, say so explicitly in `description` (e.g. 'delivery channel not stated in evidence') "
    "rather than silently omitting the distinction — the goal is that no programme entry reads "
    "as just a repeated theme word."
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

# DECISION-MAKER RULE (feedback item 8): the tool was surfacing people whose job has nothing
# to do with CSR (engineers, regional ops) just because their name appeared near CSR content
# on a page. This is now a standalone, explicit exclusion rule referenced from both the
# extraction prompt and the rubric line for decision_maker_accessibility.
DECISION_MAKER_RULE = (
    "DECISION-MAKERS — RELEVANCE FILTER (feedback priority, check this carefully): only include "
    "a person if their TITLE or the EVIDENCE CONTEXT around their name specifically ties them to "
    "CSR, sustainability, corporate foundation, community/social impact, or education/skilling "
    "partnerships. Merely being named on the same page, in the same press release, or in the "
    "same annual-report section as CSR content is NOT sufficient if their stated role is "
    "unrelated (e.g. software engineer, sales/regional head, plant operations, unrelated "
    "business-unit leadership). When uncertain whether a role qualifies, prefer to leave the "
    "person out rather than include a functionally-irrelevant name — a fundraiser needs a "
    "person they can credibly reach out to about CSR/education, and a wrong name wastes an "
    "outreach attempt and damages credibility. The CEO/MD/Chairperson may be included only if "
    "the evidence shows them personally quoted or credited on CSR/foundation matters, not by "
    "default just for holding the top role."
)

EVIDENCE_STYLE_RULE = (
    "EVERY field must trace to the evidence you were given — never invent facts, and state "
    "partial evidence as partial. Evidence/reasoning fields used for SCORING (criteria[].evidence, "
    "criteria[].reasoning, fit_rationale, alignment_rationale) are short paraphrases (under 20 "
    "words), never verbatim quotes except for exact figures, partner names, or programme names. "
    "source_excerpt fields (feedback priority: these are shown directly to the user as supporting "
    "evidence, not just used internally for scoring) may instead be a short, exact, verbatim "
    "excerpt from the source — up to about 25 words — so the user can see the actual sentence a "
    "finding is based on rather than a re-paraphrased version of it; still keep it tight and drop "
    "surrounding boilerplate."
)

HIGHLIGHT_RULE = (
    "HIGHLIGHT: in fit_rationale, alignment_rationale, delivery_model_evidence, "
    "source_quality_assessment, csr_head_note, evidence_recency, contact_pathway.channel, "
    "strategic_insight, and each criterion's evidence — bold exactly one 2-3 word "
    "decision-relevant phrase with **asterisks**. Never bold a full sentence, a lone number, or "
    "more than 3 words. Never bold names, titles, sources, URLs, booleans, or enums. Skip only "
    "if the field is empty."
)

# GEOGRAPHY RULE (feedback item 7): fundraisers need state/city level detail to know if TAP
# already operates there — country-level or vague "pan-India" mentions don't answer that.
GEOGRAPHY_RULE = (
    "GEOGRAPHIES: capture every state/city explicitly named in the evidence as a separate "
    "entry — prefer this granular level over country-level ('India') or vague scope phrases "
    "('across India', 'pan-India', 'multiple states') whenever ANY more specific place is named "
    "anywhere in the evidence, since state/city is what tells a fundraiser whether this overlaps "
    "with TAP's existing footprint. If the evidence genuinely only supports a vague scope with no "
    "state/city named anywhere, include that vague entry rather than omitting geography "
    "entirely, but do not let a vague entry substitute for specific ones that are available "
    "elsewhere in the evidence — include both if both exist."
)

FIELD_ORDER_RULE = (
    "FIELD ORDER: emit the JSON object's top-level keys in exactly the order shown in the JSON "
    "shape below, with no exceptions. This matters because your reply can be cut off by an "
    "output-length limit, and fields written first are the ones guaranteed to survive a cutoff — "
    "so short, high-value summary fields (authenticity score, source quality, evidence recency, "
    "delivery model, sector, eligibility, spend) are placed before the larger array fields "
    "(programmes, partners, decision_makers, geographies, red_flags), which are placed last "
    "since they are the most likely to be truncated safely without losing the fields other parts "
    "of the pipeline depend on."
)


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

{DECISION_MAKER_RULE}

{GEOGRAPHY_RULE}

{SOURCE_INTEGRITY_RULE}

{EVIDENCE_STYLE_RULE}

{FIELD_ORDER_RULE}

Extract, matching the JSON shape's key order exactly:
1. overall_authenticity_score (0-100) — reflects sourcing quality (primary vs secondary, how many sources actually returned usable text), not evidence volume.
2. source_quality_assessment — 1-2 sentences: primary (company/regulator) vs secondary (press/snippets) sourcing.
3. evidence_recency — one sentence on how current the evidence appears.
4. csr_head_note — one sentence, only from actual named-person context, never speculation from a bare title.
5. delivery_model (FUNDER/IMPLEMENTER/HYBRID/UNCLEAR) + delivery_model_evidence (1 sentence).
6. sector — from company-description language; UNKNOWN only if truly no clue.
7. eligibility — Section 135 applicability (LIKELY/UNLIKELY/UNKNOWN) from net worth/turnover/profit figures (kept separate from spend), plus the plain numeric business-scale fields.
8. spend — apply the SPEND VS REVENUE and EDUCATION SPEND rules strictly, including the mandatory multi-year trend search.
9. rfp_signal — an explicit call for NGO partners; default false/empty unless stated.
10. board_affinity — named board/promoter personal education-philanthropy history; default false/empty unless stated.
11. volunteering — named employee volunteering/payroll-giving touching education; default false/empty unless stated.
12. group_foundation — CSR run via a separate parent/group foundation, only if explicitly named.
13. key_facts_summary — 3-6 short bullet-style facts (as a single string, one per line prefixed with "- ") that most directly bear on education-CSR fit — this feeds directly into the scoring pass, so include anything that would move a fit judgment either up or down.
14. open_questions[] — up to 5 short, concrete, searchable items to verify.
15. programmes[] — apply the PROGRAMME rule, including the delivery-channel/beneficiary specificity requirement.
16. partners[] — apply the PARTNER rule, including similar_to_tap_profile and multi-year history.
17. decision_makers[] — apply the DECISION-MAKER relevance filter strictly; title, public_facing_score 0-100, tenure_status, linkedin_url only if a literal linkedin.com/in/ URL is present in the evidence.
18. geographies[] — apply the GEOGRAPHY rule; prefer state/city over country/vague-region entries.
19. red_flags[] — genuine contradictions or marketing-not-substance signals, severity low/medium/high. Missing/undocumented details are NOT red flags.
20. contact_pathway — the single most concrete real channel; "Not identified" if nothing exists.

Reply with ONE JSON object, nothing else, no markdown fences.

JSON shape:
{{
  "overall_authenticity_score": <int 0-100>,
  "source_quality_assessment": "<1-2 sentences>",
  "evidence_recency": "<one sentence>",
  "csr_head_note": "<one sentence>",
  "delivery_model": "<FUNDER|IMPLEMENTER|HYBRID|UNCLEAR>",
  "delivery_model_evidence": "<sentence>",
  "sector": {{"sector": "<sector>", "sub_sector": "<or empty>", "reasoning": "<short>"}},
  "eligibility": {{"plausibly_mandated": "<LIKELY|UNLIKELY|UNKNOWN>", "reasoning": "<short>", "net_worth_turnover_signal": "<short>", "net_worth_turnover_inr_crore": <number or null>, "net_profit_inr_crore": <number or null>}},
  "spend": {{"inr_crore": <number or null, education-specific only>, "display": "<exact CSR-labeled education figure or empty>", "fiscal_year": "<if stated>", "is_education_specific": <bool>, "education_pct_of_total_csr": <number or null>, "has_disclosed_budget": <bool>, "confidence": <0-100>, "source_excerpt": "<short, verbatim ok>", "trend_direction": "<RISING|FLAT|DECLINING|UNKNOWN>", "trend_evidence": "<short>", "history": [{{"fiscal_year": "<year>", "inr_crore": <number or null>, "display": "<as stated>", "source_excerpt": "<short, verbatim ok>"}}], "total_csr_inr_crore": <number or null>, "total_csr_display": "<as stated or empty>", "total_csr_fiscal_year": "<if stated>"}},
  "rfp_signal": {{"present": <bool>, "channel": "<short>", "evidence": "<short>"}},
  "board_affinity": {{"present": <bool>, "person_name": "<name or empty>", "connection": "<short>", "source_excerpt": "<short, verbatim ok>"}},
  "volunteering": {{"present": <bool>, "programme_name": "<name or empty>", "description": "<short>", "source_excerpt": "<short, verbatim ok>"}},
  "group_foundation": {{"routed_through_group": <bool>, "foundation_name": "<name or empty>", "explanation": "<short>", "source_excerpt": "<short, verbatim ok>"}},
  "key_facts_summary": "<3-6 lines, each starting with '- '>",
  "open_questions": ["<short item>", "..."],
  "programmes": [{{"name": "<exact name>", "what_is_funded": "<precise funded activity>", "beneficiary_group": "<named beneficiary group>", "beneficiary_type": "<SCHOOL_CHILDREN_CURRICULUM|ADULT|OTHER>", "description": "<short, must cover delivery channel + beneficiary + one-off-vs-ongoing per PROGRAMME rule>", "is_multi_year": <bool>, "cohort_or_scale": "<if stated>", "source_excerpt": "<short, verbatim ok>", "confidence": "<confirmed|probable>"}}],
  "partners": [{{"name": "<exact org name>", "relationship_type": "<funder|implementer|co-design|unclear>", "programme": "<or empty>", "year": "<or empty>", "geography": "<or empty>", "similar_to_tap_profile": <bool>, "source_excerpt": "<short, verbatim ok, must show relationship language>", "confidence": "<confirmed|probable>"}}],
  "decision_makers": [{{"name": "<n>", "title": "<title>", "public_facing_score": <0-100>, "tenure_status": "<NEW_UNDER_1YR|ESTABLISHED_1_3YR|ENTRENCHED_3YR_PLUS|UNKNOWN>", "tenure_evidence": "<short>", "source_excerpt": "<short, verbatim ok>", "linkedin_url": "<url or empty>"}}],
  "geographies": [{{"place": "<state/city preferred>", "source_excerpt": "<short, verbatim ok>"}}],
  "red_flags": [{{"flag": "<short label>", "severity": "<low|medium|high>", "explanation": "<short>"}}],
  "contact_pathway": {{"channel": "<sentence>", "evidence": "<short>"}}
}}"""


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

{SCORING_CONSISTENCY_RULE}

{HIGHLIGHT_RULE}

Produce, in this order:
1. criteria[] — all 17 ids below, in order, each with id, name (copy exactly as given), score 0-5, confidence 0-100, short evidence, short reasoning, drawn only from the extracted facts above. Follow the CONSISTENCY rule above for every score:
{_rubric_block()}
2. fit_score (int 0-100) — compute this yourself as the weighted average of the criteria scores you just wrote (score/5 × weight for each, summed across all 17). Do this arithmetically from your own criteria, not as a separate holistic guess, so it matches what the system independently computes from the same criteria.
3. fit_rationale (2-4 sentences): justify the scoring from the extracted facts, stating plainly what's confirmed vs inferred vs undocumented. If a named partner/programme suggests a plausible but unconfirmed entry path, you may add one sentence starting literally "Inference (unconfirmed):" naming that specific org/programme — never invent one not in the extracted facts. If decision_makers and/or partners/programmes are non-empty, end with one short sentence "Key contacts: A (Title), B (Title); Key partners: X, Y" using only names from the extracted facts. Omit that closing sentence if both lists are empty.
4. overall_semantic_alignment (0-100) + alignment_rationale (1-2 sentences) — how well the company's actual activity matches the NGO mission semantically, independent of documentation completeness.
5. strategic_insight — a 150-280 word standalone narrative (this is the lead summary shown to the user first, and should read as usable outreach material, not just an internal note): measured and evidence-grounded, leading with genuine strengths before caveats, stating plainly whether/why this is a good fit, naming strongest/weakest dimensions without dwelling on the weakest, flagging group-foundation routing if present, noting eligibility if uncertain, and giving one concrete next step. When spend is discussed, lead with the education-specific figure/trend over the total CSR figure if both are available. When a specific programme or partner is TAP-relevant, name its delivery channel explicitly (in-school/curriculum vs adult/standalone vs digital, etc.) and state concretely how TAP's own model (AI-enabled WhatsApp delivery, government-school, curriculum-embedded electives) does or doesn't overlap with it — write this so a sentence could be lifted directly into an outreach email, rather than a generic theme match like "both work in education." If TAP-similar partners exist, mention that positively. {"Since this is a screen-mode pass, if the signal is promising but sourcing is thin, say plainly that a deep-research pass would surface more (spend figures, named partners, a decision-maker) rather than treating the gap as a weakness." if mode == "screen" else ""} End with the same "Key contacts: ...; Key partners: ..." sentence format as fit_rationale (only using names from the extracted facts), omitted if both lists are empty.

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
    source_excerpt: str = Field(default="", max_length=260)


class SpendSchema(BaseModel):
    inr_crore: float | None = None
    display: str = ""
    fiscal_year: str = ""
    is_education_specific: bool = False
    education_pct_of_total_csr: float | None = None
    has_disclosed_budget: bool = False
    confidence: int = Field(ge=0, le=100, default=0)
    source_excerpt: str = Field(default="", max_length=260)
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
    description: str = Field(default="", max_length=260)
    is_multi_year: bool = False
    cohort_or_scale: str = ""
    source_excerpt: str = Field(default="", max_length=260)
    source: str = ""
    confidence: str = "confirmed"


class PartnerSchema(BaseModel):
    name: str = ""
    relationship_type: str = ""
    programme: str = Field(default="", max_length=160)
    year: str = ""
    geography: str = Field(default="", max_length=120)
    similar_to_tap_profile: bool = False
    source_excerpt: str = Field(default="", max_length=260)
    source: str = ""
    confidence: str = "confirmed"


class DecisionMakerSchema(BaseModel):
    name: str = ""
    title: str = ""
    public_facing_score: int = Field(ge=0, le=100, default=0)
    tenure_status: str = "UNKNOWN"
    tenure_evidence: str = Field(default="", max_length=200)
    source_excerpt: str = Field(default="", max_length=260)
    source: str = ""
    linkedin_url: str = ""


class GeographySchema(BaseModel):
    place: str = ""
    source_excerpt: str = Field(default="", max_length=200)
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
    source_excerpt: str = Field(default="", max_length=260)
    source: str = ""


class VolunteeringSchema(BaseModel):
    present: bool = False
    programme_name: str = ""
    description: str = Field(default="", max_length=220)
    source_excerpt: str = Field(default="", max_length=260)
    source: str = ""


class GroupFoundationSchema(BaseModel):
    routed_through_group: bool = False
    foundation_name: str = ""
    explanation: str = Field(default="", max_length=240)
    source_excerpt: str = Field(default="", max_length=260)
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
    scoring_incomplete: bool = False


async def call_anthropic_chat(
    prompt: str,
    max_tokens: int = 1400,
    temperature: float = 0.0,
    model: str | None = None,
    caller: str = "unknown",
) -> str | None:
    if not settings.anthropic_configured:
        logger.warning("anthropic call skipped caller=%s reason=not_configured", caller)
        return None

    estimated_prompt_tokens = estimate_tokens(prompt)
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
        "anthropic request caller=%s model=%s max_tokens=%d estimated_prompt_tokens=%d temperature=%.2f",
        caller, resolved_model, max_tokens, estimated_prompt_tokens, temperature,
    )

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
        return None

    if response.status_code == 429:
        logger.warning(
            "anthropic 429 caller=%s retry_after=%s body=%s",
            caller, response.headers.get("retry-after", "unknown"), response.text[:200],
        )
        return None

    if response.status_code >= 400:
        logger.error("anthropic http error caller=%s status=%d body=%s", caller, response.status_code, response.text[:400])
        return None

    try:
        body = response.json()
    except ValueError:
        logger.error("anthropic non-json response caller=%s", caller)
        return None

    logger.info("anthropic response caller=%s status=%d", caller, response.status_code)

    if body.get("stop_reason") == "max_tokens":
        logger.warning("anthropic response TRUNCATED caller=%s max_tokens=%d", caller, max_tokens)

    content_blocks = body.get("content") or []
    text_parts = [block.get("text", "") for block in content_blocks if block.get("type") == "text"]
    if not text_parts:
        logger.error("anthropic malformed response caller=%s", caller)
        return None
    return "{" + "".join(text_parts)


def parse_json_response(raw_text: str | None, expected_keys: list[str] | None = None, caller: str = "unknown") -> dict:
    if not raw_text:
        return {}
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    recovered = _recover_partial_json(cleaned)
    if recovered:
        logger.info("parse_json_response recovered via partial-json fallback caller=%s chars=%d", caller, len(cleaned))
        if expected_keys:
            missing = [key for key in expected_keys if key not in recovered]
            if missing:
                logger.warning(
                    "parse_json_response recovered object is missing expected keys caller=%s missing=%s",
                    caller, missing,
                )
        return recovered
    logger.error("parse_json_response failed to recover any JSON caller=%s chars=%d", caller, len(cleaned))
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
    """
    The deterministic weighted score (from criteria[] × CRITERIA_WEIGHTS) is always the
    source of truth for fit_score — this is the key consistency guarantee (feedback item 1):
    since criteria are now required to be evidence-anchored (see SCORING_CONSISTENCY_RULE),
    the same evidence should reliably produce the same weighted score, regardless of what
    holistic number the model separately reports. model_reported_score is used only as a
    drift signal for logging/QA, never to override the computed value.
    """
    weighted = _compute_weighted_fit_score(criteria)
    calibrated = _apply_authenticity_ceiling(weighted, authenticity_score, mode)
    final_score = int(round(max(0.0, min(100.0, calibrated))))
    if model_reported_score is not None:
        drift = abs(model_reported_score - weighted)
        if drift > 15:
            logger.warning(
                "fit_score drift flagged: model_reported=%s weighted_from_criteria=%.1f "
                "final=%d mode=%s authenticity=%d drift=%.1f — final score always uses the "
                "deterministic weighted value, this log is for prompt-quality monitoring only",
                model_reported_score, weighted, final_score, mode, authenticity_score, drift,
            )
    return final_score


def _repair_extraction(parsed: dict, caller: str = "unknown") -> dict:
    parsed = dict(parsed) if isinstance(parsed, dict) else {}

    missing_priority = [key for key in EXTRACTION_PRIORITY_KEYS if key not in parsed]
    if missing_priority:
        logger.warning(
            "_repair_extraction missing priority keys, defaults will be used caller=%s missing=%s",
            caller, missing_priority,
        )

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


def _empty_criteria() -> list[dict]:
    return [
        {
            "id": criterion_id, "name": CRITERIA_TITLES[criterion_id],
            "score": 0.0, "confidence": 0,
            "evidence": "No signal returned for this criterion", "reasoning": "", "source": "",
        }
        for criterion_id in CRITERIA_IDS
    ]


def build_extraction_only_result(extraction: dict, mode: str) -> dict:
    merged = dict(extraction)
    merged.pop("key_facts_summary", None)
    merged["criteria"] = _empty_criteria()
    merged["fit_score"] = 0
    merged["fit_rationale"] = ""
    merged["overall_semantic_alignment"] = 0
    merged["alignment_rationale"] = ""
    merged["strategic_insight"] = LLM_SCORING_UNAVAILABLE_NOTE
    merged["scoring_incomplete"] = True

    validated = _repair_analysis(merged)
    result = validated.model_dump()
    result["scoring_incomplete"] = True
    result["open_questions"] = [q.strip()[:200] for q in extraction.get("open_questions", []) if q and q.strip()][:5]

    logger.warning(
        "build_extraction_only_result company_facts_preserved mode=%r authenticity=%d "
        "partners=%d programmes=%d decision_makers=%d",
        mode, result["overall_authenticity_score"], len(result["partners"]),
        len(result["programmes"]), len(result["decision_makers"]),
    )
    return result


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
                "strategic_insight", "scoring_incomplete",
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


def anthropic_cooldown_remaining_seconds() -> float:
    return 0.0


def evidence_token_budget(company: str, mission: str, sources_manifest: str) -> int:
    scaffold_tokens = estimate_tokens(_extraction_prompt(company, mission, "", sources_manifest))
    reserved_for_output = EXTRACTION_OUTPUT_TOKEN_RESERVE
    ceiling = _anthropic_context_window() - reserved_for_output - scaffold_tokens
    return max(MIN_EVIDENCE_TOKEN_BUDGET, ceiling)


def _shrink_to_fit(company: str, mission: str, sources_manifest: str, cleaned_sources: list[dict],
                    prompt_builder, output_ceiling: int) -> tuple[str, list[dict], int]:
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
    evidence_text = combine_evidence_text(cleaned_sources)
    if not evidence_text.strip():
        logger.info("extract_company_facts skipped company=%r reason=no_evidence_text", company)
        return None

    output_ceiling = _anthropic_context_window() - EXTRACTION_OUTPUT_TOKEN_RESERVE

    def _build(evidence: str) -> str:
        return _extraction_prompt(company, mission, evidence, sources_manifest)

    evidence_text, working_sources, prompt_tokens = _shrink_to_fit(
        company, mission, sources_manifest, cleaned_sources, _build, output_ceiling,
    )

    if prompt_tokens > output_ceiling:
        logger.error(
            "extract_company_facts could not fit prompt within context window company=%r prompt_tokens=%d ceiling=%d",
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

    parsed = parse_json_response(raw_reply, expected_keys=EXTRACTION_PRIORITY_KEYS, caller=f"extract_facts:{company}")
    if not parsed:
        logger.error("extract_company_facts empty parse company=%r", company)
        return None

    extraction = _repair_extraction(parsed, caller=f"extract_facts:{company}")

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
        "extract_company_facts DONE company=%r authenticity=%d partners=%d programmes=%d decision_makers=%d red_flags=%d "
        "spend_history_years=%d geographies=%d",
        company, extraction.get("overall_authenticity_score", 0), len(extraction.get("partners", [])),
        len(extraction.get("programmes", [])), len(extraction.get("decision_makers", [])),
        len(extraction.get("red_flags", [])), len((extraction.get("spend") or {}).get("history", []) or []),
        len(extraction.get("geographies", [])),
    )
    return extraction


async def score_extracted_facts(
    company: str,
    mission: str,
    mode: str,
    extraction: dict,
    sources_manifest: str,
) -> dict | None:
    scoring_facts = {
        k: v for k, v in extraction.items()
        if k not in ("open_questions", "key_facts_summary", "overall_authenticity_score",
                      "evidence_recency", "source_quality_assessment", "csr_head_note")
    }
    prompt = _scoring_prompt(company, mission, mode, scoring_facts, sources_manifest)
    prompt_tokens = estimate_tokens(prompt)
    output_reserve = OUTPUT_TOKEN_RESERVE - EXTRACTION_OUTPUT_TOKEN_RESERVE
    output_ceiling = _anthropic_context_window() - output_reserve

    if prompt_tokens > output_ceiling:
        trimmed_facts = dict(scoring_facts)
        for list_field in ("programmes", "partners", "decision_makers", "geographies", "red_flags"):
            if trimmed_facts.get(list_field):
                trimmed_facts[list_field] = trimmed_facts[list_field][:5]
        prompt = _scoring_prompt(company, mission, mode, trimmed_facts, sources_manifest)
        prompt_tokens = estimate_tokens(prompt)

    if prompt_tokens > output_ceiling:
        logger.error(
            "score_extracted_facts could not fit prompt within context window company=%r prompt_tokens=%d ceiling=%d",
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

    parsed = parse_json_response(raw_reply, expected_keys=["criteria", "fit_score"], caller=f"score_facts:{company}")
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
    extraction = await extract_company_facts(company, mission, cleaned_sources, sources_manifest)
    if not extraction:
        return None

    scoring = await score_extracted_facts(company, mission, mode, extraction, sources_manifest)
    if not scoring:
        logger.error(
            "analyze_and_score_company scoring pass failed after successful extraction, "
            "returning extraction-only result company=%r mode=%r",
            company, mode,
        )
        return build_extraction_only_result(extraction, mode)

    merged = dict(extraction)
    merged.pop("key_facts_summary", None)
    merged["criteria"] = scoring.get("criteria", [])
    merged["fit_score"] = scoring.get("fit_score", 0)
    merged["fit_rationale"] = scoring.get("fit_rationale", "")
    merged["overall_semantic_alignment"] = scoring.get("overall_semantic_alignment", 0)
    merged["alignment_rationale"] = scoring.get("alignment_rationale", "")
    merged["strategic_insight"] = scoring.get("strategic_insight", "")
    merged["scoring_incomplete"] = False

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
