import functools
import json
import logging
import os
import re
import time
import typing

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.config import settings
from app.pipeline.decision_reconciliation import reconcile_extraction
from app.pipeline.textproc import combine_evidence_text, estimate_tokens

logger = logging.getLogger("tap.llm")

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"
ANTHROPIC_PROMPT_CACHING_BETA_HEADER = "prompt-caching-2024-07-31"

LLM_UNAVAILABLE_EVIDENCE = "LLM unavailable — unable to generate evidence"
LLM_SCORING_UNAVAILABLE_NOTE = (
    "Automated scoring could not complete for this run, but the facts below were "
    "successfully extracted from the fetched sources — verify and score manually."
)

EXTRACTION_OUTPUT_TOKEN_RESERVE = 4200
SCORING_OUTPUT_TOKEN_RESERVE = 4200
MIN_EVIDENCE_TOKEN_BUDGET = 500
ANTHROPIC_REQUEST_TIMEOUT_SECONDS = 120.0
MIN_PROMPT_TRIM_CHARS = 150
MAX_PROMPT_SHRINK_ATTEMPTS = 6
PROMPT_SHRINK_SAFETY_MARGIN = 120
DEFAULT_ANTHROPIC_CONTEXT_WINDOW = 200000
DEFAULT_LLM_DUMP_DIR = "/tmp/fundfinder_llm_dumps"
MIN_CACHEABLE_BLOCK_TOKENS = 1024

ANTHROPIC_DEFAULT_COOLDOWN_SECONDS = 60.0

_anthropic_cooldown_until = 0.0
_http_client: httpx.AsyncClient | None = None

EXTRACTION_PRIORITY_KEYS = [
    "overall_authenticity_score",
    "source_quality_assessment",
    "evidence_recency",
    "delivery_model",
    "delivery_model_evidence",
    "sector",
    "eligibility",
    "spend",
    "entity_structure",
]

SEARCH_DIRECTIVE_PRIORITIES = ("HIGH", "MEDIUM", "LOW")
MAX_SEARCH_DIRECTIVES = 5


def _verbose_logging_enabled() -> bool:
    return bool(getattr(settings, "verbose_pipeline_logging", True))


def _llm_dump_dir() -> str:
    return getattr(settings, "llm_dump_dir", DEFAULT_LLM_DUMP_DIR)


def _dump_llm_io(caller: str, kind: str, content: str) -> None:
    if not _verbose_logging_enabled():
        return
    try:
        dump_dir = _llm_dump_dir()
        os.makedirs(dump_dir, exist_ok=True)
        safe_caller = re.sub(r"[^a-zA-Z0-9_.:-]", "_", caller)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(dump_dir, f"{timestamp}_{safe_caller}_{kind}.txt")
        with open(path, "w", encoding="utf-8") as dump_file:
            dump_file.write(content)
        logger.info("llm io dumped caller=%s kind=%s path=%s bytes=%d", caller, kind, path, len(content))
    except Exception as exc:
        logger.warning("failed to dump llm io caller=%s kind=%s error=%s", caller, kind, exc)


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=ANTHROPIC_REQUEST_TIMEOUT_SECONDS)
    return _http_client


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


@functools.lru_cache(maxsize=1)
def _rubric_block() -> str:
    return "\n".join(f"- {key}: {value}" for key, value in _RUBRIC.items())


@functools.lru_cache(maxsize=1)
def _criteria_json_template() -> str:
    return ",\n".join(
        f'    {{"id": "{cid}", "name": "{CRITERIA_TITLES[cid]}", "score": <0-5 or null>, "confidence": <0-100>, "evidence": "<short paraphrase>", "reasoning": "<short>"}}'
        for cid in CRITERIA_IDS
    )


MODE_CALIBRATION = {
    "screen": {
        "stance": (
            "MODE: SCREEN (triage pass, fewer sources than deep by design). Judge whether "
            "{company} deserves a deep-research pass, not a final verdict. Read thin-but-promising "
            "signals generously — clear education/CSR activity with no disqualifying red flag should "
            "score as 'worth a deep dive' even without spend figures, named partners, or a named "
            "decision-maker yet. Reserve low scores for an actual negative signal (no CSR at all, "
            "explicitly non-education CSR, a stated policy against NGO partnerships) — thin sourcing "
            "alone is never sufficient reason for a low score here."
        ),
        "authenticity_cap_threshold": 15,
        "authenticity_cap_ceiling": 78,
        "rule_detail": "short",
    },
    "deep": {
        "stance": (
            "MODE: DEEP RESEARCH (outreach-ready brief, informs an actual outreach decision). Hold "
            "evidence to a stricter standard than a triage pass — undocumented should still read "
            "generously per the philosophy below, but claims should be well-grounded since a person "
            "may act on this directly."
        ),
        "authenticity_cap_threshold": 30,
        "authenticity_cap_ceiling": 65,
        "rule_detail": "full",
    },
}


def _mode_calibration(mode: str) -> dict:
    return MODE_CALIBRATION.get(mode, MODE_CALIBRATION["deep"])


CACHEABLE_EXTRACTION_RULES = (
    "EVIDENCE_STATE_RULE: every numeric or categorical field carries an explicit state: FOUND "
    "(a source states this value), NOT_FOUND_IN_SOURCE (sources don't mention it), or "
    "CONFIRMED_ABSENT (a source positively states zero/nonexistent). If NOT_FOUND_IN_SOURCE, "
    "the value field is null. Never write 0, 0.0, \"none\" or \"nil\" for something simply not "
    "found — zero is a finding, absence of a finding is not zero, and the two must render "
    "differently. A value cannot be FOUND without a citable source number.\n\n"
    "TREND_RULE: never describe a trend in words. Emit each multi-year point as "
    "{fiscal_year/year, value, source} oldest-to-newest plus a series_state "
    "(FOUND/NOT_FOUND_IN_SOURCE/CONFIRMED_ABSENT). Leave trend_direction UNKNOWN in every field you "
    "write — the application derives RISING/FLAT/DECLINING from the series, never take that label "
    "from you. No relative-time phrases (\"eight years ago\"); absolute years only. Fewer than two "
    "points is not a trend — emit the points, leave trend_direction UNKNOWN.\n\n"
    "ENTITY_STRUCTURE_RULE: before programmes/partners, scan for every distinct named legal vehicle "
    "connected to the company that could run or fund CSR — parent/global company, a separately "
    "incorporated India entity or branch, any separately-named foundation/trust/CSR arm (e.g. 'X "
    "Foundation', 'X India Private Limited', 'X AG Mumbai Branch'). Populate entity_structure only "
    "with layers actually named in evidence; leave a field empty rather than guess. Do not assume a "
    "parent and a same-branded foundation are the same funding channel unless evidence says so — every "
    "programme/partner must set funded_by_entity to whichever named entity evidence actually "
    "attributes it to, never silently defaulted to the parent name. If a candidate entity string "
    "reads as two different companies concatenated (e.g. a search snippet mashing two unrelated "
    "mentions together), it is not a real entity — only use names that read as one coherent org.\n\n"
    "SPEND_VS_REVENUE_RULE: revenue, turnover, net worth, profit, market cap, EBITDA describe "
    "business SCALE, never CSR spend — never place them in spend.display/inr_crore, never call them "
    "CSR spend/budget/fund. spend.has_disclosed_budget=true only for a figure explicitly labeled CSR "
    "expenditure/spend/budget, or a stated CSR-mandate percentage applied to a stated profit figure; "
    "otherwise false and inr_crore=0. Business-scale figures go only in "
    "eligibility.net_worth_turnover_signal / net_worth_turnover_inr_crore / net_profit_inr_crore, "
    "never in spend.\n\n"
    "EDUCATION_SPEND_RULE: spend.inr_crore/display/fiscal_year/trend_* hold ONLY the "
    "education-specific slice — set is_education_specific=true and populate these only when evidence "
    "states an education-specific figure or percentage. A total-CSR-only figure goes in "
    "total_csr_inr_crore/display/fiscal_year, clearly labeled total, never as the education budget. "
    "Three distinct outcomes, never collapsed into one another: (1) education-specific figure found "
    "-> populate, state FOUND; (2) total CSR only, no split -> leave education fields empty, state "
    "NOT_FOUND_IN_SOURCE, note that only a total was available; (3) source states no education "
    "funding -> inr_crore=0, state CONFIRMED_ABSENT. Outcome 2 is the common case and must never "
    "render as outcome 3. Actively scan for figures from more than one fiscal year — reports often "
    "disclose 2-3 years side by side even when only the latest is prominently mentioned. Populate "
    "spend.history[] with every distinct (fiscal_year, figure) pair found; treat finding the trend as "
    "equally important as finding the headline number, but never label the direction yourself.\n\n"
    "PROFIT_HISTORY_RULE: scan for net profit / profit after tax figures across multiple fiscal "
    "years the same way you scan for CSR spend trend. Populate eligibility.net_profit_history[] with "
    "every distinct (fiscal_year, net_profit_inr_crore) pair, each with a verbatim source_excerpt. Set "
    "net_profit_trend_direction to RISING/FLAT/DECLINING only with two-plus years, else UNKNOWN. This "
    "feeds a downstream Section 135 obligation calculation — do not compute it yourself, only extract "
    "profit figures faithfully.\n\n"
    "PARTNER_RULE: include a named third-party org as a partner only if evidence shows an actual "
    "relationship (funds, co-designs, implements with, partners with, delivers via, works with). "
    "confidence='confirmed' if the relationship verb is explicit; 'probable' if named alongside the "
    "company in a CSR/education context but the language is vague/implied. Exclude internal campaign "
    "names, generic unnamed government references, award/certifying bodies. Fill programme/year/"
    "geography only where evidence states them. Set similar_to_tap_profile=true if the partner is "
    "itself an education/skilling/government-school-facing NGO or intermediary (even if unrelated to "
    "TAP) — a genuine positive signal for partnership_quality and delivery_model_fit. Any third-party "
    "org named in your own prose fields must also appear in partners[] — never describe a partnership "
    "in prose while leaving it out of the structured list. Actively check evidence for partner "
    "mentions across different fiscal years / annual reports, and include each distinct (partner, "
    "year) combination as its own entry rather than collapsing multi-year partners into one undated "
    "entry. A Development Impact Bond, outcomes fund, flagship-named programme, or any other "
    "formally-named initiative must be extracted as its own programme entry (confidence='probable' "
    "at minimum) the instant it's referenced anywhere, even in a single sentence or footnote-like "
    "mention — never folded into a generic theme sentence instead. Such initiatives typically involve "
    "several delivery partners — look for every organisation named alongside it and add each as a "
    "separate partner entry rather than citing only the most prominent name.\n\n"
    "TAP_PRIORITY_RULE: rank every programme by relevance to the mission — school education, STEM, "
    "AI, coding, digital skills, government schools, students and teachers are priority areas. List "
    "every priority-area programme first and in full: name, what it does, beneficiaries, implementing "
    "partner, geography and scale. Other CSR programmes (health, environment, non-schooling "
    "livelihoods, employee wellness, etc.) follow briefly, one line each, no chain-completeness "
    "expectation. If a priority-area programme is named anywhere — even a passing mention, a "
    "footnote, a phrase buried in an unrelated paragraph — it must appear in programmes[]. Never drop "
    "or compress a TAP-relevant programme to make room for a less relevant one, and never let a "
    "generic theme mention substitute for a named priority-area programme evidence actually contains "
    "elsewhere. Before finalizing programmes[], re-scan raw evidence specifically for priority-area "
    "keywords (STEM, coding, AI, digital, government school, Atal Tinkering Lab, curriculum, teacher "
    "training, girls in tech) and confirm every resulting mention has a corresponding entry — a "
    "keyword hit with no matching entry is a missed extraction, not a genuine absence.\n\n"
    "PROGRAMME_RULE: include a named programme only if you can state what is funded and who "
    "benefits, at whatever specificity evidence actually gives (ordinary phrasing like 'government-"
    "school students' is fine). A bare theme mention with no name, no beneficiary, no funded "
    "activity should be left out rather than inflated into a full entry. confidence='confirmed' with "
    "one additional concrete supporting detail (scale, duration, since-when) beyond name/what's-"
    "funded/beneficiary; 'probable' otherwise. Any specific initiative named in your own prose fields "
    "must also appear in programmes[]. For every programme, `description` must state wherever "
    "evidence allows: (a) delivery channel (in-school/curriculum-embedded, standalone adult workshop, "
    "digital/app-based, vocational, or other), (b) concrete beneficiary (school-going children vs "
    "out-of-school youth vs adults vs teachers), (c) one-off vs ongoing — a theme word alone ('life "
    "skills') is not useful since the same theme can mean opposite things in practice. If evidence "
    "truly doesn't support (a)-(c), say so explicitly in `description` rather than silently omitting "
    "the distinction. Set chain_missing_elements to whichever of "
    "[beneficiaries, geography, partner, government_school_involvement, scale_or_outcomes, "
    "funding_amount] evidence genuinely does not state anywhere for that programme — never guess a "
    "value just to leave the list empty; be precise about what's actually missing versus what you "
    "found elsewhere and should have already filled in.\n\n"
    "DECISION_MAKER_RULE: include a person only if their TITLE or the evidence context around their "
    "name specifically ties them to CSR, sustainability, corporate foundation, community/social "
    "impact, or education/skilling partnerships — being merely named on the same page or in the same "
    "press release as CSR content is not sufficient if their stated role is unrelated (software "
    "engineer, sales/regional head, plant operations, unrelated business-unit leadership). When "
    "uncertain, prefer to leave the person out. The CEO/MD/Chairperson may be included only if "
    "evidence shows them personally quoted or credited on CSR/foundation matters, never by default "
    "for holding the top role. Any person appearing in a source named 'people_search' already passed "
    "a currency and CSR-relevance check upstream — transcribe them faithfully (name, title, "
    "source_excerpt), set is_india_specific from what the excerpt says; only exclude one of these if "
    "its own excerpt contains explicit former-role language (Previously, Formerly, Former, ex-, Past, "
    "or a stated end year) the upstream check may have missed. Source-of-truth order (fall to the "
    "next only when the one above yields nothing): (1) the company's own leadership/CSR-team page, "
    "(2) a signed foreword or signatory in the annual/CSR report, (3) a press release naming the "
    "person in role, (4) a LinkedIn search-result snippet — parse the snippet, never attempt to fetch "
    "the profile page itself (authwall). A snippet-only contact is UNVERIFIED — label it via "
    "tenure_status/is_india_specific plus a tenure_evidence note. CURRENCY TEST: reject as current any "
    "headline/snippet whose former-role language (Previously, Formerly, Former, ex-, Past, a stated "
    "end year) sits close to and describes THIS person's role; a bare year range or former-role "
    "keyword describing something else elsewhere in the same excerpt does not disqualify them. THREE-"
    "CHECK VERIFICATION before including anyone: (1) still at the company per the currency test, (2) "
    "designation as stated is CSR/foundation/sustainability/philanthropy-relevant, not inferred from "
    "context alone, (3) role genuinely covers this company's CSR function, not an unrelated "
    "department. INDIA PRIORITY: when both an India-based and a global-level contact surface for the "
    "same company, include both if evidence supports each, but set is_india_specific=true only for "
    "the one whose title/scope is explicitly India-focused; a global CEO quoted in a press release is "
    "a citation, not a route in. If nobody passes the currency test, return an empty list — returning "
    "nobody is a usable result, a former employee shown as current is worse than nobody. "
    "csr_head_note and decision_makers[] describe the same fact and must never contradict each other: "
    "if anyone appears anywhere with a currency-test-passing CSR/foundation/sustainability title, add "
    "them to decision_makers[] AND reference them by name in csr_head_note — only state no CSR head "
    "was identified if decision_makers[] is genuinely empty after every rule above.\n\n"
    "GEOGRAPHY_RULE: capture every state/city explicitly named as its own entry — prefer this over "
    "country-level ('India') or vague scope ('across India', 'pan-India') whenever any more specific "
    "place is named anywhere in evidence. If evidence genuinely only supports a vague scope with no "
    "state/city named anywhere, include that vague entry rather than omit geography, but never let a "
    "vague entry substitute for specific ones available elsewhere — include both if both exist.\n\n"
    "SOURCE_INTEGRITY_RULE: before treating a fragment as evidence, confirm it's a genuine "
    "descriptive sentence about the company's own activity, not a nav menu, link list, or heading "
    "run-on, and confirm it's actually about the company being analysed and not a different entity "
    "sharing the page (especially for people-search/LinkedIn snippets — a profile whose employer is a "
    "different company is not evidence about this company even if this company's name appears "
    "elsewhere on the page). A person's individual career history describes that person, never a "
    "company programme. Some companies operate more than one distinct legal vehicle for CSR (a direct "
    "India branch AND a separately-named foundation) — these are related but not interchangeable; for "
    "every programme/partner/spend figure, note which specific named entity evidence actually "
    "attributes it to whenever that's clear, and set funded_by_entity accordingly; if evidence is "
    "ambiguous about which entity is responsible, say so rather than assume they're the same org.\n\n"
    "EVIDENCE_STYLE_RULE: every field must trace to the evidence given — never invent facts, state "
    "partial evidence as partial. Scoring-facing evidence/reasoning fields (criteria[].evidence, "
    "criteria[].reasoning, fit_rationale, alignment_rationale) are short paraphrases under 20 words, "
    "never verbatim quotes except exact figures/partner names/programme names. source_excerpt fields "
    "are shown directly to the user as supporting evidence — these may instead be a short, exact, "
    "verbatim excerpt up to about 25 words so the user sees the actual sentence a finding rests on, "
    "still dropping surrounding boilerplate.\n\n"
    "SEARCH_DIRECTIVE_RULE: for up to 5 genuinely open questions, emit an object with `question` "
    "(short, human-readable), `search_query` (a concrete string to type directly into a search engine "
    "— not a restatement of the question, specific enough to plausibly return the missing fact: "
    "company name plus the precise missing detail, e.g. company name plus a named programme plus "
    "\"geography\" or \"beneficiaries\", or company name plus \"CSR spend\" plus a fiscal year), "
    "`target_field` (the dotted path into this JSON shape the answer would fill, e.g. "
    "\"spend.history\", \"programmes[].geography\", \"decision_makers\", "
    "\"entity_structure.foundation_entity\" — leave empty if no single field applies; when leaving it "
    "empty is genuinely unavoidable, use a short unique slug built from the question itself so the "
    "directive still has a stable identity downstream, e.g. \"open:named_ngo_partner_scale\"), and "
    "`priority` (HIGH/MEDIUM/LOW, reflecting how much the missing fact would change scoring). Only a "
    "question genuinely unresolved in the evidence given — this is a to-do list for a second research "
    "pass, not general commentary.\n\n"
    "CROSS_SECTION_CONSISTENCY_RULE (apply this to everything above, read it last): every fact stated "
    "in a narrative/summary field (csr_head_note, key_facts_summary, delivery_model_evidence, "
    "fit_rationale, strategic_insight) must also exist in the corresponding structured array "
    "(decision_makers[], programmes[], partners[]). Naming a person or programme in prose while "
    "leaving them out of, or contradicting, the structured list is a critical error — downstream "
    "reports and spreadsheets read only from the structured arrays, so anything true only in prose is "
    "invisible everywhere else, and anything contradicted in prose looks fabricated. Before "
    "finalizing your reply, re-read every narrative field and confirm each named person and programme "
    "has a matching structured entry, and that no narrative field asserts an absence ('no CSR head "
    "identified', 'no named programmes found') that the structured arrays contradict."
)

CACHEABLE_SCORING_RULES = (
    "SCORING PHILOSOPHY: most CSR activity in India is only partially documented online. Silence "
    "about something is not evidence against it. Never score a criterion at 0, and never treat a "
    "company as a poor fit, purely because a fact wasn't surfaced by the sources given — reserve 0 "
    "only for evidence that actively contradicts fit. Where the record is quiet on a criterion, score "
    "it from sector, scale, and adjacent CSR behavior visible elsewhere in evidence, and label it in "
    "`evidence` as an inferred estimate, not a confirmed absence. When genuinely torn between two "
    "adjacent scores, prefer the higher one — undocumented should never read as a negative signal. "
    "This is per-criterion; it does not mean invent facts, and it does not mean treat every company as "
    "a great fit — evidence that actively points away from fit should bring scores down honestly, "
    "just as evidence supporting fit should bring them up. A low fit score and low research confidence "
    "are NOT the same conclusion: if evidence is thin, say so plainly and let the evidence-coverage "
    "gate (applied outside this prompt) handle whether the score is shown at all — do not pre-"
    "emptively collapse toward a low score just because sources are sparse.\n\n"
    "You may decline to score. If a criterion has no evidence in the extracted facts, emit score: "
    "null with confidence: 0 and a one-line reason naming what was missing in `evidence`. Do not "
    "guess a low number to fill the slot — a null score is not a bad score, it's the honest output "
    "when research didn't reach that criterion, and the application excludes it from the average "
    "rather than let it drag the average down. Scoring 1/5 because you found nothing is damaging "
    "because it's indistinguishable downstream from a company genuinely assessed and found weak. "
    "Score 0 only where a source actively contradicts the criterion. For any criterion scored null, "
    "if you can see a concrete, ready-to-run search query that would plausibly resolve it, add it to "
    "unscored_criteria_search_directives per SEARCH_DIRECTIVE_RULE — this is how the gap gets "
    "followed up on, not just recorded.\n\n"
    "CONSISTENCY: scores must be repeatable — the same evidence scored again should land on the same "
    "numbers. Never assign a score from general impression, reputation, or brand size. For every "
    "criterion, first identify the single most relevant fact/quote in extracted evidence, map that "
    "fact to a score using the rubric line for that criterion, and write that fact (not a vibe) into "
    "`evidence`. If two criteria could plausibly take the same score for the same underlying reason, "
    "they should — don't vary scores across similar criteria without a distinct evidentiary reason for "
    "each. Don't let overall enthusiasm about the company inflate individual scores beyond what each "
    "one's own evidence supports.\n\n"
    "CONFIDENCE_SEPARATION_RULE: you produce two independent judgements — do not let one bleed into "
    "the other. fit_score is how well evidence you DO have indicates alignment; research_coverage is "
    "how much of the picture you retrieved at all. A company can be a strong fit on thin evidence, or "
    "a poor fit on thorough evidence — the report must say both. Do not compute fit_score yourself; "
    "emit the criteria array and stop — the application computes the weighted average from what you "
    "scored, and research_coverage from what you declined. Emitting your own fit_score reintroduces "
    "exactly the blending this rule prevents. In fit_rationale, never explain a low score by saying "
    "evidence was limited — if evidence was limited, the affected criteria should be null and the "
    "rationale should speak only to what was actually found.\n\n"
    "HIGHLIGHT: in fit_rationale, alignment_rationale, delivery_model_evidence, "
    "source_quality_assessment, csr_head_note, evidence_recency, contact_pathway.channel, "
    "strategic_insight, and each criterion's evidence — bold exactly one 2-3 word decision-relevant "
    "phrase with **asterisks**. Never bold a full sentence, a lone number, or more than 3 words. Never "
    "bold names, titles, sources, URLs, booleans, or enums. Skip only if the field is empty.\n\n"
    "SEARCH_DIRECTIVE_RULE: for up to 5 genuinely open questions, emit an object with `question` "
    "(short, human-readable), `search_query` (a concrete string to type directly into a search engine "
    "— not a restatement of the question, specific enough to plausibly return the missing fact), "
    "`target_field` (the criterion id this would resolve — for the scoring pass this must be one of "
    "the 17 criteria ids, never empty and never a dotted evidence path), and `priority` "
    "(HIGH/MEDIUM/LOW). Only a question genuinely unresolved in the evidence given.\n\n"
    "CROSS_SECTION_CONSISTENCY_RULE: every fact named in a narrative field must be consistent with "
    "the structured facts already given to you — never assert in prose an absence ('no CSR head "
    "identified', 'no named programmes found') that the extracted facts contradict."
)


def _extraction_prompt_static_block() -> str:
    return CACHEABLE_EXTRACTION_RULES


def _extraction_prompt_dynamic_block(company: str, mission: str, evidence_text: str, sources_manifest: str) -> str:
    return f"""You are a meticulous fact-extraction analyst. Extract every concrete, sourced fact about {company}'s India CSR activity from the evidence below. Do NOT score or judge fit — that happens in a separate pass. Your only job here is complete, accurate, well-cited extraction.

NGO MISSION (context only, for judging what counts as education-relevant — do not score against it here): {mission}

EVIDENCE (from sources actually fetched for {company} — numbered sources below can be cited):
\"\"\"
{evidence_text}
\"\"\"

SOURCES:
{sources_manifest}

Extract, matching the JSON shape's key order exactly:
1. overall_authenticity_score (0-100) — reflects sourcing quality (primary vs secondary, how many sources actually returned usable text), not evidence volume.
2. source_quality_assessment — 1-2 sentences: primary (company/regulator) vs secondary (press/snippets) sourcing.
3. evidence_recency — one sentence on how current the evidence appears.
4. csr_head_note — one sentence, only from actual named-person context, never speculation from a bare title. Must agree with decision_makers[] per DECISION_MAKER_RULE.
5. delivery_model (FUNDER/IMPLEMENTER/HYBRID/UNCLEAR) + delivery_model_evidence (1 sentence).
6. sector — from company-description language; UNKNOWN only if truly no clue.
7. eligibility — Section 135 applicability (LIKELY/UNLIKELY/UNKNOWN) from net worth/turnover/profit figures (kept separate from spend), plus plain numeric business-scale fields, plus the profit history required by PROFIT_HISTORY_RULE.
8. spend — apply SPEND_VS_REVENUE_RULE and EDUCATION_SPEND_RULE strictly, including the mandatory multi-year trend search, and apply EVIDENCE_STATE_RULE / TREND_RULE to every figure and history point.
9. entity_structure — apply ENTITY_STRUCTURE_RULE; leave any layer empty rather than guessing.
10. rfp_signal — an explicit call for NGO partners; default false/empty unless stated.
11. board_affinity — named board/promoter personal education-philanthropy history; default false/empty unless stated.
12. volunteering — named employee volunteering/payroll-giving touching education; default false/empty unless stated.
13. group_foundation — CSR run via a separate parent/group foundation, only if explicitly named.
14. key_facts_summary — 3-6 short bullet-style facts (single string, one per line prefixed "- ") that most directly bear on education-CSR fit — feeds the scoring pass, include anything that would move a fit judgment either way. Never use this as a dumping ground for a programme that belongs in programmes[] instead.
15. search_directives[] — apply SEARCH_DIRECTIVE_RULE; up to 5 entries.
16. programmes[] — apply PROGRAMME_RULE and TAP_PRIORITY_RULE together: priority-area programmes listed first and in full, then everything else briefly. Apply delivery-channel/beneficiary specificity and chain-completeness fields to every entry.
17. partners[] — apply PARTNER_RULE, including similar_to_tap_profile, multi-year history, and named-format initiatives.
18. decision_makers[] — apply DECISION_MAKER_RULE's source-of-truth order, currency test, three-check verification, and consistency sub-rule strictly; linkedin_url only if a literal linkedin.com/in/ URL is present in evidence. Every people_search hit that passes the currency test must appear here.
19. geographies[] — apply GEOGRAPHY_RULE; prefer state/city over country/vague-region entries.
20. red_flags[] — genuine contradictions or marketing-not-substance signals, severity low/medium/high. Missing/undocumented details are NOT red flags.
21. contact_pathway — the single most concrete real channel; "Not identified" if nothing exists.

7b. eligibility.net_profit_history — apply PROFIT_HISTORY_RULE; source 10 (multi_year_financials), when present, is specifically curated for side-by-side year figures — check it first for profit_history and spend.history.

Before replying, run the CROSS_SECTION_CONSISTENCY_RULE check once over your own draft.

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
  "eligibility": {{"plausibly_mandated": "<LIKELY|UNLIKELY|UNKNOWN>", "reasoning": "<short>", "net_worth_turnover_signal": "<short>", "net_worth_turnover_inr_crore": <number, 0 if unknown>, "net_profit_inr_crore": <number, 0 if unknown>, "net_profit_fiscal_year": "<if stated>", "net_profit_history": [{{"fiscal_year": "<year>", "net_profit_inr_crore": <number, 0 if unknown>, "source_excerpt": "<short, verbatim ok>"}}], "net_profit_trend_direction": "<RISING|FLAT|DECLINING|UNKNOWN>"}},
  "spend": {{"inr_crore": <number, 0 if unknown, education-specific only>, "display": "<exact CSR-labeled education figure or empty>", "fiscal_year": "<if stated>", "is_education_specific": <bool>, "education_pct_of_total_csr": <number, 0 if unknown>, "has_disclosed_budget": <bool>, "confidence": <0-100>, "source_excerpt": "<short, verbatim ok>", "state": "<FOUND|NOT_FOUND_IN_SOURCE|CONFIRMED_ABSENT>", "start_year": "<four-digit year if stated, else empty>", "trend_direction": "UNKNOWN", "trend_evidence": "<short>", "series_state": "<FOUND|NOT_FOUND_IN_SOURCE|CONFIRMED_ABSENT>", "history": [{{"fiscal_year": "<year>", "inr_crore": <number, 0 if unknown>, "display": "<as stated>", "source_excerpt": "<short, verbatim ok>"}}], "total_csr_inr_crore": <number, 0 if unknown>, "total_csr_display": "<as stated or empty>", "total_csr_fiscal_year": "<if stated>"}},
  "entity_structure": {{"parent_company": "<name or empty>", "india_entity": "<name or empty>", "foundation_entity": "<name or empty>", "notes": "<short, only if evidence clarifies how these relate>"}},
  "rfp_signal": {{"present": <bool>, "channel": "<short>", "evidence": "<short>"}},
  "board_affinity": {{"present": <bool>, "person_name": "<name or empty>", "connection": "<short>", "source_excerpt": "<short, verbatim ok>"}},
  "volunteering": {{"present": <bool>, "programme_name": "<name or empty>", "description": "<short>", "source_excerpt": "<short, verbatim ok>"}},
  "group_foundation": {{"routed_through_group": <bool>, "foundation_name": "<name or empty>", "explanation": "<short>", "source_excerpt": "<short, verbatim ok>"}},
  "key_facts_summary": "<3-6 lines, each starting with '- '>",
  "search_directives": [{{"question": "<short item>", "search_query": "<concrete ready-to-run search query>", "target_field": "<dotted path or empty>", "priority": "<HIGH|MEDIUM|LOW>"}}],
  "programmes": [{{"name": "<exact name>", "what_is_funded": "<precise funded activity>", "beneficiary_group": "<named beneficiary group>", "beneficiary_type": "<SCHOOL_CHILDREN_CURRICULUM|ADULT|OTHER>", "description": "<short, must cover delivery channel + beneficiary + one-off-vs-ongoing per PROGRAMME_RULE>", "is_multi_year": <bool>, "cohort_or_scale": "<if stated>", "funded_by_entity": "<name from entity_structure or empty>", "chain_missing_elements": ["<subset of beneficiaries|geography|partner|government_school_involvement|scale_or_outcomes|funding_amount>"], "source_excerpt": "<short, verbatim ok>", "confidence": "<confirmed|probable>"}}],
  "partners": [{{"name": "<exact org name>", "relationship_type": "<funder|implementer|co-design|unclear>", "programme": "<or empty>", "year": "<or empty>", "geography": "<or empty>", "similar_to_tap_profile": <bool>, "funded_by_entity": "<name from entity_structure or empty>", "source_excerpt": "<short, verbatim ok, must show relationship language>", "confidence": "<confirmed|probable>"}}],
  "decision_makers": [{{"name": "<n>", "title": "<title>", "public_facing_score": <0-100>, "tenure_status": "<NEW_UNDER_1YR|ESTABLISHED_1_3YR|ENTRENCHED_3YR_PLUS|UNKNOWN>", "tenure_evidence": "<short>", "is_india_specific": <bool>, "source_excerpt": "<short, verbatim ok>", "linkedin_url": "<url or empty>"}}],
  "geographies": [{{"place": "<state/city preferred>", "source_excerpt": "<short, verbatim ok>"}}],
  "red_flags": [{{"flag": "<short label>", "severity": "<low|medium|high>", "explanation": "<short>"}}],
  "contact_pathway": {{"channel": "<sentence>", "evidence": "<short>"}}
}}"""


def _extraction_prompt_blocks(company: str, mission: str, evidence_text: str, sources_manifest: str) -> list[dict]:
    return [
        {"type": "text", "text": _extraction_prompt_static_block()},
        {"type": "text", "text": _extraction_prompt_dynamic_block(company, mission, evidence_text, sources_manifest)},
    ]


def _extraction_prompt(company: str, mission: str, evidence_text: str, sources_manifest: str) -> str:
    return "\n\n".join(block["text"] for block in _extraction_prompt_blocks(company, mission, evidence_text, sources_manifest))


def _scoring_prompt_static_block() -> str:
    return CACHEABLE_SCORING_RULES


def _scoring_prompt_dynamic_block(company: str, mission: str, mode: str, extraction: dict, sources_manifest: str,
                                   csr_obligation: dict | None = None) -> str:
    calibration = _mode_calibration(mode)
    extraction_json = json.dumps(extraction, ensure_ascii=False, indent=2)

    obligation_block = ""
    if csr_obligation and csr_obligation.get("computable"):
        obligation_block = f"""
CSR OBLIGATION SIGNAL (pre-computed, do not recalculate — just weave into narrative if relevant):
{json.dumps(csr_obligation, ensure_ascii=False)}
A company with latest_year_underspending=true has compelled, unplaced CSR funds — mention this
explicitly and favorably in strategic_insight if present, as it is a stronger warmth signal than
profitability alone. Do not fabricate this signal if computable=false.
"""

    return f"""You are a careful, fair-minded CSR partnerships analyst judging whether {company} is a good funding/partnership fit for an Indian education NGO. A separate extraction pass already pulled every fact below from the fetched evidence — do not re-extract or add new facts, only score against what's here and write the narrative fields.

{calibration['stance']}

NGO MISSION: {mission}

EXTRACTED FACTS FOR {company} (already verified against evidence — numbered sources below can be cited):
\"\"\"
{extraction_json}
\"\"\"

SOURCES:
{sources_manifest}
{obligation_block}
Produce, in this order:
1. criteria[] — all 17 ids below, in order, each with id, name (copy exactly as given), score 0-5 or null, confidence 0-100, short evidence, short reasoning, drawn only from the extracted facts above. Follow CONSISTENCY for every score. `confidence` must reflect how directly the extracted facts support THIS criterion specifically — not overall company confidence, and not another criterion's confidence. A criterion resting on an inferred/sector-default judgment (per SCORING PHILOSOPHY) should carry materially lower confidence than one resting on an explicit, named fact. If a criterion has no evidence at all, emit score: null, confidence: 0, and say what was missing in `evidence` — per SCORING PHILOSOPHY, do not guess a low number just to fill the slot:
{_rubric_block()}
2. Do NOT compute or emit fit_score yourself — per CONFIDENCE_SEPARATION_RULE, the application computes it deterministically from the criteria array above. Stop after criteria and move directly to the narrative fields below.
3. fit_rationale (2-4 sentences): justify the scoring from the extracted facts, stating plainly what's confirmed vs inferred vs undocumented. Never explain a low score by citing limited evidence — a criterion with limited evidence should be null, not low. If a named partner/programme suggests a plausible but unconfirmed entry path, you may add one sentence starting literally "Inference (unconfirmed):" naming that specific org/programme — never invent one not in the extracted facts. If decision_makers and/or partners/programmes are non-empty, end with one short sentence "Key contacts: A (Title), B (Title); Key partners: X, Y" using only names from the extracted facts — never write a sentence implying no contact or programme was found if either array is non-empty. Omit that closing sentence only if both lists are empty.
4. overall_semantic_alignment (0-100) + alignment_rationale (1-2 sentences) — how well the company's actual activity matches the NGO mission semantically, independent of documentation completeness.
5. strategic_insight — a 150-280 word standalone narrative (this is the lead summary shown to the user first, and should read as usable outreach material, not just an internal note): measured and evidence-grounded, leading with genuine strengths before caveats, stating plainly whether/why this is a good fit, naming strongest/weakest dimensions without dwelling on the weakest, flagging group-foundation routing if present, noting eligibility if uncertain, weaving in the CSR obligation signal above if present, and giving one concrete next step. Lead with priority-area programmes (school education, STEM, AI, coding, digital skills, government schools) over generic CSR themes when both exist in the extracted facts. When spend is discussed, lead with the education-specific figure/trend over the total CSR figure if both are available, and never state a trend word unless the spend series actually supports it (per TREND_RULE) — if only one year of spend is known, describe the single figure and do not claim a trend. When a specific programme or partner is TAP-relevant, name its delivery channel explicitly (in-school/curriculum vs adult/standalone vs digital, etc.) and state concretely how TAP's own model (AI-enabled WhatsApp delivery, government-school, curriculum-embedded electives) does or doesn't overlap with it — write this so a sentence could be lifted directly into an outreach email, rather than a generic theme match like "both work in education." If TAP-similar partners exist, mention that positively. {"Since this is a screen-mode pass, if the signal is promising but sourcing is thin, say plainly that a deep-research pass would surface more (spend figures, named partners, a decision-maker) rather than treating the gap as a weakness." if mode == "screen" else ""} End with the same "Key contacts: ...; Key partners: ..." sentence format as fit_rationale (only using names from the extracted facts, and never contradicting a non-empty decision_makers/partners/programmes list), omitted only if both lists are empty.
6. unscored_criteria_search_directives[] — apply SEARCH_DIRECTIVE_RULE, but populate this only for criteria you scored null above, with target_field set to that criterion's id. Never populate this for a criterion you actually scored.

All criteria ids appear exactly once, in the order listed, each with its name copied exactly as given above. Keep every string concise so the full reply fits comfortably in your output budget.

Reply with ONE JSON object, nothing else, no markdown fences.

JSON shape:
{{
  "criteria": [
{_criteria_json_template()}
  ],
  "fit_rationale": "<2-4 sentences, one **2-3 word** highlight, optional Inference/Key-contacts clauses>",
  "overall_semantic_alignment": <int 0-100>,
  "alignment_rationale": "<1-2 sentences, one **2-3 word** highlight>",
  "strategic_insight": "<150-280 word narrative, one **2-3 word** highlight, optional Inference/Key-contacts clauses>",
  "unscored_criteria_search_directives": [{{"question": "<short item>", "search_query": "<concrete ready-to-run search query>", "target_field": "<criterion id>", "priority": "<HIGH|MEDIUM|LOW>"}}]
}}"""


def _scoring_prompt_blocks(company: str, mission: str, mode: str, extraction: dict, sources_manifest: str,
                            csr_obligation: dict | None = None) -> list[dict]:
    return [
        {"type": "text", "text": _scoring_prompt_static_block()},
        {"type": "text", "text": _scoring_prompt_dynamic_block(company, mission, mode, extraction, sources_manifest, csr_obligation)},
    ]


def _scoring_prompt(company: str, mission: str, mode: str, extraction: dict, sources_manifest: str,
                     csr_obligation: dict | None = None) -> str:
    return "\n\n".join(
        block["text"]
        for block in _scoring_prompt_blocks(company, mission, mode, extraction, sources_manifest, csr_obligation)
    )


class CriterionResultSchema(BaseModel):
    id: str
    name: str = ""
    score: float | None = Field(ge=0, le=5, default=None)
    confidence: int = Field(ge=0, le=100, default=0)
    evidence: str = Field(default="", max_length=240)
    reasoning: str = Field(default="", max_length=240)
    source: str = Field(default="")


class SpendYearSchema(BaseModel):
    fiscal_year: str = ""
    inr_crore: float = 0.0
    display: str = ""
    source: str = ""
    source_excerpt: str = Field(default="", max_length=260)


class SpendSchema(BaseModel):
    inr_crore: float = 0.0
    display: str = ""
    fiscal_year: str = ""
    is_education_specific: bool = False
    education_pct_of_total_csr: float = 0.0
    has_disclosed_budget: bool = False
    confidence: int = Field(ge=0, le=100, default=0)
    source_excerpt: str = Field(default="", max_length=260)
    source: str = ""
    state: str = "NOT_FOUND_IN_SOURCE"
    start_year: str = ""
    trend_direction: str = "UNKNOWN"
    trend_evidence: str = Field(default="", max_length=240)
    trend_source: str = ""
    series_state: str = "NOT_FOUND_IN_SOURCE"
    history: list[SpendYearSchema] = Field(default_factory=list)
    total_csr_inr_crore: float = 0.0
    total_csr_display: str = ""
    total_csr_fiscal_year: str = ""
    estimated_min_inr_crore: float = 0.0
    estimated_basis: str = Field(default="", max_length=200)
    estimated_is_computed: bool = False


class EntityStructureSchema(BaseModel):
    parent_company: str = Field(default="", max_length=160)
    india_entity: str = Field(default="", max_length=160)
    foundation_entity: str = Field(default="", max_length=160)
    notes: str = Field(default="", max_length=280)


PROGRAMME_CHAIN_ELEMENTS = frozenset([
    "beneficiaries", "geography", "partner", "government_school_involvement",
    "scale_or_outcomes", "funding_amount",
])

EVIDENCE_STATES = frozenset(["FOUND", "NOT_FOUND_IN_SOURCE", "CONFIRMED_ABSENT"])
TREND_DIRECTIONS = frozenset(["RISING", "FLAT", "DECLINING", "UNKNOWN"])


class ProgrammeSchema(BaseModel):
    name: str = ""
    what_is_funded: str = Field(default="", max_length=200)
    beneficiary_group: str = Field(default="", max_length=160)
    beneficiary_type: str = "OTHER"
    description: str = Field(default="", max_length=260)
    is_multi_year: bool = False
    cohort_or_scale: str = ""
    funded_by_entity: str = Field(default="", max_length=160)
    chain_missing_elements: list[str] = Field(default_factory=list)
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
    funded_by_entity: str = Field(default="", max_length=160)
    source_excerpt: str = Field(default="", max_length=260)
    source: str = ""
    confidence: str = "confirmed"


class DecisionMakerSchema(BaseModel):
    name: str = ""
    title: str = ""
    public_facing_score: int = Field(ge=0, le=100, default=0)
    tenure_status: str = "UNKNOWN"
    tenure_evidence: str = Field(default="", max_length=200)
    is_india_specific: bool = False
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


class NetProfitYearSchema(BaseModel):
    fiscal_year: str = ""
    net_profit_inr_crore: float = 0.0
    source_excerpt: str = Field(default="", max_length=260)
    source: str = ""


class EligibilitySchema(BaseModel):
    plausibly_mandated: str = "UNKNOWN"
    reasoning: str = Field(default="", max_length=280)
    net_worth_turnover_signal: str = Field(default="", max_length=200)
    net_worth_turnover_inr_crore: float = 0.0
    net_profit_inr_crore: float = 0.0
    net_profit_fiscal_year: str = ""
    net_profit_history: list[NetProfitYearSchema] = Field(default_factory=list)
    net_profit_trend_direction: str = "UNKNOWN"
    source: str = ""


class SectorSchema(BaseModel):
    sector: str = "UNKNOWN"
    sub_sector: str = ""
    reasoning: str = Field(default="", max_length=200)


class CsrObligationYearSchema(BaseModel):
    fiscal_year: str = ""
    net_profit_inr_crore: float = 0.0
    statutory_obligation_inr_crore: float = 0.0
    actual_total_csr_inr_crore: float = 0.0
    shortfall_inr_crore: float = 0.0
    is_underspending: bool = False


class CsrObligationSignalSchema(BaseModel):
    computable: bool = False
    reason_not_computable: str = Field(default="", max_length=200)
    years: list[CsrObligationYearSchema] = Field(default_factory=list)
    latest_year_underspending: bool = False
    profit_trend_direction: str = "UNKNOWN"
    explanation: str = Field(default="", max_length=320)


class SearchDirectiveSchema(BaseModel):
    question: str = Field(default="", max_length=240)
    search_query: str = Field(default="", max_length=300)
    target_field: str = Field(default="", max_length=160)
    priority: str = "MEDIUM"


class FullAnalysisSchema(BaseModel):
    fit_score: int | None = Field(ge=0, le=100, default=None)
    research_coverage: int = Field(ge=0, le=100, default=0)
    fit_rationale: str = Field(default="", max_length=600)
    overall_semantic_alignment: int = Field(ge=0, le=100, default=0)
    alignment_rationale: str = Field(default="", max_length=500)
    delivery_model: str = "UNCLEAR"
    delivery_model_evidence: str = Field(default="", max_length=220)
    delivery_model_source: str = ""
    spend: SpendSchema = SpendSchema()
    entity_structure: EntityStructureSchema = EntityStructureSchema()
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
    search_directives: list[SearchDirectiveSchema] = Field(default_factory=list)
    unscored_criteria_search_directives: list[SearchDirectiveSchema] = Field(default_factory=list)
    strategic_insight: str = Field(default="", max_length=2200)
    scoring_incomplete: bool = False
    csr_obligation_signal: CsrObligationSignalSchema = CsrObligationSignalSchema()


SECTION_135_CSR_RATE = 0.02
MIN_PROFIT_YEARS_FOR_TREND = 2


def _derive_trend_direction(history: list[dict], value_key: str = "inr_crore") -> str:
    valid_points = [
        entry for entry in (history or [])
        if isinstance(entry, dict) and entry.get("fiscal_year") and entry.get(value_key, 0)
    ]
    if len(valid_points) < 2:
        return "UNKNOWN"
    ordered = sorted(valid_points, key=lambda e: e.get("fiscal_year", ""))
    first_value = ordered[0].get(value_key, 0)
    last_value = ordered[-1].get(value_key, 0)
    if first_value <= 0:
        return "UNKNOWN"
    if last_value > first_value * 1.05:
        return "RISING"
    if last_value < first_value * 0.95:
        return "DECLINING"
    return "FLAT"


def _apply_derived_spend_trend(spend: dict) -> dict:
    spend = dict(spend or {})
    history = spend.get("history") or []
    if len(history) >= 2:
        spend["trend_direction"] = _derive_trend_direction(history, value_key="inr_crore")
        spend["series_state"] = "FOUND"
    elif len(history) == 1:
        spend["trend_direction"] = "UNKNOWN"
        if spend.get("series_state") not in EVIDENCE_STATES:
            spend["series_state"] = "FOUND"
    else:
        spend["trend_direction"] = "UNKNOWN"
        if spend.get("series_state") not in EVIDENCE_STATES:
            spend["series_state"] = "NOT_FOUND_IN_SOURCE"

    if spend.get("state") not in EVIDENCE_STATES:
        if spend.get("is_education_specific") and (spend.get("inr_crore") or spend.get("display")):
            spend["state"] = "FOUND"
        elif spend.get("is_education_specific") and spend.get("inr_crore", 0) == 0 and spend.get("display", "") == "":
            spend["state"] = "NOT_FOUND_IN_SOURCE"
        else:
            spend["state"] = "NOT_FOUND_IN_SOURCE"

    if spend.get("state") == "NOT_FOUND_IN_SOURCE":
        spend["inr_crore"] = 0.0
        spend["display"] = spend.get("display", "") if spend.get("is_education_specific") is False else ""

    return spend


def _apply_derived_profit_trend(eligibility: dict) -> dict:
    eligibility = dict(eligibility or {})
    history = eligibility.get("net_profit_history") or []
    if len(history) >= 2:
        eligibility["net_profit_trend_direction"] = _derive_trend_direction(
            history, value_key="net_profit_inr_crore"
        )
    elif eligibility.get("net_profit_trend_direction") not in TREND_DIRECTIONS:
        eligibility["net_profit_trend_direction"] = "UNKNOWN"
    elif len(history) < 2 and eligibility.get("net_profit_trend_direction") != "UNKNOWN":
        eligibility["net_profit_trend_direction"] = "UNKNOWN"
    return eligibility


def compute_csr_obligation_signal(eligibility: dict, spend: dict) -> dict:
    profit_years = eligibility.get("net_profit_history") or []
    valid_profit_years = [
        y for y in profit_years
        if isinstance(y, dict) and y.get("fiscal_year") and y.get("net_profit_inr_crore", 0) > 0
    ]

    if not valid_profit_years:
        return {
            "computable": False,
            "reason_not_computable": "No disclosed net profit figure was found in evidence.",
            "years": [],
            "latest_year_underspending": False,
            "profit_trend_direction": "UNKNOWN",
            "explanation": "",
        }

    spend_by_year = {}
    for entry in (spend.get("history") or []):
        fy = entry.get("fiscal_year", "")
        if fy:
            spend_by_year[fy] = entry.get("inr_crore", 0.0)
    if spend.get("total_csr_fiscal_year") and spend.get("total_csr_inr_crore"):
        spend_by_year.setdefault(spend["total_csr_fiscal_year"], spend["total_csr_inr_crore"])
    if spend.get("fiscal_year") and spend.get("inr_crore") and spend.get("is_education_specific") is False:
        spend_by_year.setdefault(spend["fiscal_year"], spend["inr_crore"])

    computed_years = []
    for entry in sorted(valid_profit_years, key=lambda y: y.get("fiscal_year", "")):
        fy = entry["fiscal_year"]
        profit = entry["net_profit_inr_crore"]
        obligation = round(profit * SECTION_135_CSR_RATE, 2)
        actual = spend_by_year.get(fy, 0.0)
        shortfall = round(max(0.0, obligation - actual), 2)
        computed_years.append({
            "fiscal_year": fy,
            "net_profit_inr_crore": profit,
            "statutory_obligation_inr_crore": obligation,
            "actual_total_csr_inr_crore": actual,
            "shortfall_inr_crore": shortfall,
            "is_underspending": actual > 0 and shortfall > 0,
        })

    latest = computed_years[-1] if computed_years else None
    latest_underspending = bool(latest and latest["is_underspending"])

    trend = eligibility.get("net_profit_trend_direction", "UNKNOWN")
    if trend == "UNKNOWN" and len(valid_profit_years) >= MIN_PROFIT_YEARS_FOR_TREND:
        profits_in_order = [y["net_profit_inr_crore"] for y in sorted(valid_profit_years, key=lambda y: y["fiscal_year"])]
        if profits_in_order[-1] > profits_in_order[0] * 1.05:
            trend = "RISING"
        elif profits_in_order[-1] < profits_in_order[0] * 0.95:
            trend = "DECLINING"
        else:
            trend = "FLAT"

    explanation = ""
    if latest_underspending:
        explanation = (
            f"In {latest['fiscal_year']}, statutory CSR obligation was approximately "
            f"₹{latest['statutory_obligation_inr_crore']:.1f} crore (2% of ₹{latest['net_profit_inr_crore']:.1f} crore "
            f"net profit) against actual CSR spend of ₹{latest['actual_total_csr_inr_crore']:.1f} crore — "
            f"a shortfall of approximately ₹{latest['shortfall_inr_crore']:.1f} crore. Companies underspending "
            f"against their statutory obligation carry compelled, unplaced CSR funds, which can be a stronger "
            f"funding signal than profitability alone."
        )
    elif latest and not latest["actual_total_csr_inr_crore"]:
        explanation = (
            f"Statutory obligation for {latest['fiscal_year']} is estimable at approximately "
            f"₹{latest['statutory_obligation_inr_crore']:.1f} crore, but no matching actual CSR spend figure "
            f"for that year was found in evidence, so underspend cannot be confirmed."
        )

    return {
        "computable": True,
        "reason_not_computable": "",
        "years": computed_years,
        "latest_year_underspending": latest_underspending,
        "profit_trend_direction": trend,
        "explanation": explanation,
    }


async def call_anthropic_chat(
    prompt: str | list[dict],
    max_tokens: int = 1400,
    temperature: float = 0.0,
    model: str | None = None,
    caller: str = "unknown",
    use_prompt_caching: bool = False,
) -> str | None:
    if not settings.anthropic_configured:
        logger.warning("anthropic call skipped caller=%s reason=not_configured", caller)
        return None

    if isinstance(prompt, str):
        user_content: str | list[dict] = prompt
        full_prompt_text_for_estimate = prompt
    else:
        blocks = list(prompt)
        if use_prompt_caching and blocks:
            static_tokens = estimate_tokens(blocks[0].get("text", ""))
            if static_tokens >= MIN_CACHEABLE_BLOCK_TOKENS:
                blocks[0] = {**blocks[0], "cache_control": {"type": "ephemeral"}}
        user_content = blocks
        full_prompt_text_for_estimate = "\n\n".join(b.get("text", "") for b in blocks)

    estimated_prompt_tokens = estimate_tokens(full_prompt_text_for_estimate)
    resolved_model = model or settings.anthropic_model
    payload = {
        "model": resolved_model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "{"},
        ],
    }

    headers = {
        "x-api-key": settings.anthropic_api_key,
        "anthropic-version": ANTHROPIC_API_VERSION,
        "Content-Type": "application/json",
    }
    if use_prompt_caching:
        headers["anthropic-beta"] = ANTHROPIC_PROMPT_CACHING_BETA_HEADER

    logger.info(
        "anthropic request caller=%s model=%s max_tokens=%d estimated_prompt_tokens=%d temperature=%.2f caching=%s",
        caller, resolved_model, max_tokens, estimated_prompt_tokens, temperature, use_prompt_caching,
    )
    _dump_llm_io(caller, "prompt", full_prompt_text_for_estimate)
    logger.debug("anthropic FULL PROMPT caller=%s\n%s", caller, full_prompt_text_for_estimate)

    try:
        client = _get_http_client()
        response = await client.post(ANTHROPIC_MESSAGES_URL, headers=headers, json=payload)
    except httpx.HTTPError as exc:
        logger.error("anthropic transport error caller=%s error=%s", caller, exc)
        return None

    if response.status_code == 429:
        retry_after_header = response.headers.get("retry-after", "")
        try:
            retry_after_seconds = float(retry_after_header)
        except (TypeError, ValueError):
            retry_after_seconds = ANTHROPIC_DEFAULT_COOLDOWN_SECONDS
        global _anthropic_cooldown_until
        _anthropic_cooldown_until = max(
            _anthropic_cooldown_until, time.monotonic() + max(0.0, retry_after_seconds)
        )
        logger.warning(
            "anthropic 429 caller=%s retry_after=%s cooldown_until_monotonic=%.1f body=%s",
            caller, retry_after_header or "unknown", _anthropic_cooldown_until, response.text[:200],
        )
        return None

    if response.status_code >= 400:
        logger.error("anthropic http error caller=%s status=%d body=%s", caller, response.status_code, response.text[:400])
        return None

    try:
        body = response.json()
    except ValueError:
        logger.error("anthropic non-json response caller=%s raw_text=%r", caller, response.text[:2000])
        return None

    usage = body.get("usage") or {}
    logger.info(
        "anthropic response caller=%s status=%d stop_reason=%s input_tokens=%s output_tokens=%s "
        "cache_creation_input_tokens=%s cache_read_input_tokens=%s",
        caller, response.status_code, body.get("stop_reason"),
        usage.get("input_tokens"), usage.get("output_tokens"),
        usage.get("cache_creation_input_tokens"), usage.get("cache_read_input_tokens"),
    )

    if body.get("stop_reason") == "max_tokens":
        logger.warning("anthropic response TRUNCATED caller=%s max_tokens=%d", caller, max_tokens)

    content_blocks = body.get("content") or []
    text_parts = [block.get("text", "") for block in content_blocks if block.get("type") == "text"]
    if not text_parts:
        logger.error("anthropic malformed response caller=%s raw_body=%s", caller, json.dumps(body)[:2000])
        return None
    full_reply = "{" + "".join(text_parts)
    _dump_llm_io(caller, "response", full_reply)
    logger.debug("anthropic FULL RESPONSE caller=%s\n%s", caller, full_reply)
    return full_reply


def parse_json_response(raw_text: str | None, expected_keys: list[str] | None = None, caller: str = "unknown") -> dict:
    if not raw_text:
        logger.warning("parse_json_response called with empty raw_text caller=%s", caller)
        return {}
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            logger.info(
                "parse_json_response OK caller=%s top_level_keys=%s",
                caller, list(parsed.keys()),
            )
            return parsed
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning(
            "parse_json_response direct json.loads failed caller=%s error=%s chars=%d",
            caller, exc, len(cleaned),
        )
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
    logger.error("parse_json_response failed to recover any JSON caller=%s chars=%d raw_preview=%r", caller, len(cleaned), cleaned[:500])
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
_FORMER_ROLE_KEYWORD_PATTERN = re.compile(
    r"\b(previously|formerly|former|ex-|past)\b", re.IGNORECASE,
)
_FORMER_ROLE_YEAR_RANGE_PATTERN = re.compile(
    r"\b(19|20)\d{2}\s*[-–—]\s*(present|now|\d{4})\b", re.IGNORECASE,
)
_ROLE_INDICATOR_WORD_PATTERN = re.compile(
    r"\b(role|position|served|was|as|title|designation)\b", re.IGNORECASE,
)
_FORMER_ROLE_PROXIMITY_WINDOW_CHARS = 50


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


def _former_role_language_near_role_context(text: str) -> str:
    if not text:
        return ""
    for match in _FORMER_ROLE_KEYWORD_PATTERN.finditer(text):
        window_start = max(0, match.start() - _FORMER_ROLE_PROXIMITY_WINDOW_CHARS)
        window_end = min(len(text), match.end() + _FORMER_ROLE_PROXIMITY_WINDOW_CHARS)
        window = text[window_start:window_end]
        if _ROLE_INDICATOR_WORD_PATTERN.search(window):
            return match.group(0)
    for match in _FORMER_ROLE_YEAR_RANGE_PATTERN.finditer(text):
        window_start = max(0, match.start() - _FORMER_ROLE_PROXIMITY_WINDOW_CHARS)
        window_end = min(len(text), match.end() + _FORMER_ROLE_PROXIMITY_WINDOW_CHARS)
        window = text[window_start:window_end]
        if _ROLE_INDICATOR_WORD_PATTERN.search(window):
            return match.group(0)
    return ""


def _looks_like_former_role_with_match(title: str = "", tenure_evidence: str = "",
                                        source_excerpt: str = "") -> tuple[bool, str, str]:
    if title:
        match = _FORMER_ROLE_KEYWORD_PATTERN.search(title) or _FORMER_ROLE_YEAR_RANGE_PATTERN.search(title)
        if match:
            return True, match.group(0), "title"
    for field_name, text in (("tenure_evidence", tenure_evidence), ("source_excerpt", source_excerpt)):
        matched = _former_role_language_near_role_context(text)
        if matched:
            return True, matched, field_name
    return False, "", ""


def _looks_like_former_role(*texts: str) -> bool:
    title = texts[0] if len(texts) > 0 else ""
    tenure_evidence = texts[1] if len(texts) > 1 else ""
    source_excerpt = texts[2] if len(texts) > 2 else ""
    matched, _, _ = _looks_like_former_role_with_match(title, tenure_evidence, source_excerpt)
    return matched


_NARRATIVE_CSR_TITLE_PATTERN = re.compile(
    r"\b(csr|corporate social responsibility|esg|sustainability|corporate responsibility|"
    r"social impact|community relations|philanthrop|csr\s*&?\s*esg|foundation|"
    r"diversity|inclusion|d&i|di&e|responsible business|citizenship|purpose)\b.{0,50}"
    r"\b(lead|leader|head|director|manager|officer|committee|partner|chair|"
    r"vp|vice\s*president|avp|svp|evp|trustee|chief)\b"
    r"|\b(head|director|lead|leader|partner|chair|manager|vp|vice\s*president|"
    r"avp|svp|evp|trustee|chief|officer)\b.{0,50}"
    r"\b(csr|corporate social responsibility|esg|sustainability|corporate responsibility|"
    r"social impact|community relations|philanthrop|foundation|diversity|inclusion|"
    r"responsible business|citizenship|purpose)\b",
    re.IGNORECASE,
)

_NARRATIVE_NAME_PATTERN = re.compile(
    r"\b((?:Dr\.?\s+|Mr\.?\s+|Ms\.?\s+|Mrs\.?\s+)?[A-Z][a-zA-Z.'-]*(?:\s+[A-Z][a-zA-Z.'-]*){1,4})\b"
)

_NARRATIVE_NAME_STOPWORDS = {
    "CGI", "CSR", "ESG", "India", "TAP", "NGO", "MCD", "BMC", "SCERT",
    "CEO", "MD", "The", "This", "That", "Dr", "Mr", "Ms", "Mrs",
}


def _looks_like_person_name_token_run(candidate: str) -> bool:
    tokens = [t for t in re.sub(r"^(Dr|Mr|Ms|Mrs)\.?\s+", "", candidate).split() if t]
    core_tokens = [t.rstrip(".") for t in tokens]
    if not (1 < len(core_tokens) <= 4):
        return False
    if any(tok in _NARRATIVE_NAME_STOPWORDS for tok in core_tokens):
        return False
    return all(tok[0].isupper() for tok in core_tokens if tok)


def _extract_named_people_from_narrative(*texts: str) -> list[dict]:
    found = []
    seen_keys = set()
    for text in texts:
        if not text:
            continue
        for match in _NARRATIVE_NAME_PATTERN.finditer(text):
            candidate = match.group(1).strip()
            if not _looks_like_person_name_token_run(candidate):
                continue
            window_start = max(0, match.start() - 90)
            window_end = min(len(text), match.end() + 90)
            window = text[window_start:window_end]
            if not _NARRATIVE_CSR_TITLE_PATTERN.search(window):
                continue
            if _looks_like_former_role(window):
                continue
            key = "".join(ch for ch in candidate.lower() if ch.isalnum())
            if key in seen_keys:
                continue
            seen_keys.add(key)
            found.append({
                "name": candidate,
                "title": "",
                "source_excerpt": window.strip()[:260],
            })
    return found


def _reconcile_narrative_named_people_into_decision_makers(extraction: dict, caller: str = "unknown") -> dict:
    decision_makers = extraction.get("decision_makers") or []
    existing_keys = {
        "".join(ch for ch in (person.get("name", "") or "").lower() if ch.isalnum())
        for person in decision_makers
    }

    narrative_fields = (
        extraction.get("csr_head_note", ""),
        extraction.get("key_facts_summary", ""),
        extraction.get("delivery_model_evidence", ""),
        extraction.get("fit_rationale", ""),
        extraction.get("strategic_insight", ""),
    )
    narrative_people = _extract_named_people_from_narrative(*narrative_fields)

    added_names = []
    for person in narrative_people:
        key = "".join(ch for ch in person["name"].lower() if ch.isalnum())
        if not key or key in existing_keys:
            continue
        existing_keys.add(key)
        decision_makers.append({
            "name": person["name"],
            "title": person.get("title", ""),
            "public_facing_score": 45,
            "tenure_status": "UNKNOWN",
            "tenure_evidence": "Named in narrative evidence text; not independently structured or dated.",
            "is_india_specific": False,
            "source_excerpt": person.get("source_excerpt", "")[:260],
            "source": "",
            "linkedin_url": "",
        })
        added_names.append(person["name"])

    if added_names:
        extraction["decision_makers"] = decision_makers
        logger.info(
            "_reconcile_narrative_named_people_into_decision_makers recovered names not in structured "
            "array caller=%s names=%s", caller, added_names,
        )
    return extraction


_NO_CSR_HEAD_CONTRADICTION_PATTERN = re.compile(
    r"no\s+(?:named\s+)?(?:csr|sustainability)\s+head\s+(?:has\s+been\s+)?identified|"
    r"no\s+csr\s+head\s+found|no\s+decision[\s-]makers?\s+found|"
    r"no\s+named\s+programmes?\s+(?:found|identified)|"
    r"insufficient\s+(?:evidence|data)\s+(?:on|for)\s+decision[\s-]makers?",
    re.IGNORECASE,
)


def _enforce_decision_maker_narrative_consistency(extraction: dict) -> dict:
    decision_makers = extraction.get("decision_makers") or []
    narrative_fields = (
        "csr_head_note", "key_facts_summary", "delivery_model_evidence",
        "fit_rationale", "strategic_insight",
    )
    if decision_makers:
        lead = decision_makers[0]
        lead_line = f"{lead.get('name', '')} — {lead.get('title', '')}".strip(" —")
        for field in narrative_fields:
            value = extraction.get(field, "")
            if isinstance(value, str) and value and _NO_CSR_HEAD_CONTRADICTION_PATTERN.search(value):
                if field == "csr_head_note":
                    extraction[field] = lead_line
                else:
                    extraction[field] = _NO_CSR_HEAD_CONTRADICTION_PATTERN.sub("", value).strip()
    else:
        note = extraction.get("csr_head_note", "")
        if isinstance(note, str) and note and _NO_CSR_HEAD_CONTRADICTION_PATTERN.search(note):
            extraction["csr_head_note"] = ""
    return extraction


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


def _is_optional_field(field) -> bool:
    args = typing.get_args(field.annotation)
    return bool(args) and type(None) in args


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
        if value is None and _is_optional_field(field):
            return None
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            fallback = field.default
            if value is None and _is_optional_field(field):
                return None
            value = fallback if isinstance(fallback, (int, float)) and not isinstance(fallback, bool) else 0
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


def _sanitize_chain_missing_elements(values) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned = []
    for value in values:
        if isinstance(value, str) and value.strip() in PROGRAMME_CHAIN_ELEMENTS:
            if value.strip() not in cleaned:
                cleaned.append(value.strip())
    return cleaned


def _normalize_search_directive_priority(value: str) -> str:
    normalized = (value or "").strip().upper()
    return normalized if normalized in SEARCH_DIRECTIVE_PRIORITIES else "MEDIUM"


def _build_auto_search_query(company: str, question: str) -> str:
    question = (question or "").strip()
    if not question:
        return ""
    return f'"{company}" {question}'.strip()


def _slugify_directive_question(question: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in (question or "").strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return f"open:{normalized[:60]}" if normalized else ""


def _sanitize_search_directives(company: str, raw_value, cap: int = MAX_SEARCH_DIRECTIVES,
                                 restrict_target_field_to: set | None = None) -> list[dict]:
    if not isinstance(raw_value, list):
        return []
    directives = []
    for entry in raw_value:
        if isinstance(entry, str):
            if restrict_target_field_to is not None:
                continue
            question = entry.strip()
            if not question:
                continue
            target_field = _slugify_directive_question(question)
            directives.append({
                "question": question[:240],
                "search_query": _build_auto_search_query(company, question)[:300],
                "target_field": target_field,
                "priority": "MEDIUM",
            })
        elif isinstance(entry, dict):
            question = str(entry.get("question", "")).strip()
            search_query = str(entry.get("search_query", "")).strip()
            if not question and not search_query:
                continue
            if not search_query:
                search_query = _build_auto_search_query(company, question)
            target_field = str(entry.get("target_field", "")).strip()[:160]
            if restrict_target_field_to is not None:
                if target_field not in restrict_target_field_to:
                    continue
            elif not target_field:
                target_field = _slugify_directive_question(question)
            directives.append({
                "question": question[:240],
                "search_query": search_query[:300],
                "target_field": target_field,
                "priority": _normalize_search_directive_priority(entry.get("priority", "")),
            })
        if len(directives) >= cap:
            break
    return directives


def _scored_criteria(criteria: list[dict]) -> list[dict]:
    return [c for c in criteria if c.get("score") is not None]


def _compute_weighted_fit_score(criteria: list[dict]) -> float | None:
    scored = _scored_criteria(criteria)
    if not scored:
        return None
    total_weight = 0.0
    weighted_sum = 0.0
    for entry in scored:
        criterion_id = entry.get("id", "")
        weight = CRITERIA_WEIGHTS.get(criterion_id)
        if weight is None:
            continue
        score_0_to_5 = max(0.0, min(5.0, float(entry.get("score", 0) or 0)))
        weighted_sum += (score_0_to_5 / 5.0) * weight
        total_weight += weight
    if total_weight == 0:
        return None
    return (weighted_sum / total_weight) * 100.0


def compute_research_coverage(criteria: list[dict]) -> int:
    if not criteria:
        return 0
    scored = _scored_criteria(criteria)
    return round(100 * len(scored) / len(criteria))


def _apply_authenticity_ceiling(fit_score: float, authenticity_score: int, mode: str) -> float:
    calibration = _mode_calibration(mode)
    threshold = calibration["authenticity_cap_threshold"]
    ceiling = calibration["authenticity_cap_ceiling"]
    if authenticity_score < threshold and fit_score > ceiling:
        return float(ceiling)
    return fit_score


def compute_final_fit_score(criteria: list[dict], authenticity_score: int, mode: str,
                             model_reported_score: int | None = None, company: str = "") -> int | None:
    weighted = _compute_weighted_fit_score(criteria)
    if weighted is None:
        logger.info(
            "compute_final_fit_score company=%r mode=%s no_scored_criteria fit_score=None", company, mode,
        )
        return None
    calibrated = _apply_authenticity_ceiling(weighted, authenticity_score, mode)
    final_score = int(round(max(0.0, min(100.0, calibrated))))
    logger.info(
        "mode_comparison_debug company=%r mode=%s weighted_pre_ceiling=%.1f ceiling_applied=%s final=%d",
        company, mode, weighted, calibrated != weighted, final_score,
    )
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


LOW_COVERAGE_AUTHENTICITY_THRESHOLD = 30
LOW_COVERAGE_CRITERIA_CONFIDENCE_THRESHOLD = 45
LOW_COVERAGE_HIGH_WEIGHT_FLOOR = 8
LOW_COVERAGE_HIGH_WEIGHT_CONFIDENCE_THRESHOLD = 25

LOW_COVERAGE_PLAIN_AVERAGE_THRESHOLD = 45
LOW_COVERAGE_MIN_CONFIDENT_CRITERIA_COUNT = 6
LOW_COVERAGE_CONFIDENT_CRITERION_THRESHOLD = 40
LOW_COVERAGE_MIN_CONFIDENT_WEIGHT_SHARE = 0.45
LOW_COVERAGE_LOW_CONFIDENCE_CRITERION_THRESHOLD = 25
LOW_COVERAGE_MAX_LOW_CONFIDENCE_SHARE = 0.5

RESEARCH_CONFIDENCE_HIGH_AUTHENTICITY = 65
RESEARCH_CONFIDENCE_HIGH_WEIGHTED_CONF = 65
RESEARCH_CONFIDENCE_MEDIUM_AUTHENTICITY = 40
RESEARCH_CONFIDENCE_MEDIUM_WEIGHTED_CONF = 45


def average_criteria_confidence(criteria: list[dict]) -> float:
    if not criteria:
        return 0.0
    return sum(c.get("confidence", 0) for c in criteria) / len(criteria)


def weighted_average_criteria_confidence(criteria: list[dict]) -> float:
    if not criteria:
        return 0.0
    total_weight = 0.0
    weighted_sum = 0.0
    for entry in criteria:
        weight = CRITERIA_WEIGHTS.get(entry.get("id", ""))
        if weight is None:
            continue
        confidence = max(0, min(100, int(entry.get("confidence", 0) or 0)))
        weighted_sum += confidence * weight
        total_weight += weight
    if total_weight == 0:
        return 0.0
    return weighted_sum / total_weight


def _low_confidence_high_weight_criterion(criteria: list[dict]) -> dict | None:
    worst = None
    for entry in criteria:
        weight = CRITERIA_WEIGHTS.get(entry.get("id", ""))
        if weight is None or weight < LOW_COVERAGE_HIGH_WEIGHT_FLOOR:
            continue
        confidence = max(0, min(100, int(entry.get("confidence", 0) or 0)))
        if confidence >= LOW_COVERAGE_HIGH_WEIGHT_CONFIDENCE_THRESHOLD:
            continue
        if worst is None or confidence < worst.get("confidence", 0):
            worst = entry
    return worst


def _confident_weight_share(criteria: list[dict], threshold: int) -> tuple[float, int]:
    total_weight = 0.0
    confident_weight = 0.0
    confident_count = 0
    for entry in criteria:
        weight = CRITERIA_WEIGHTS.get(entry.get("id", ""))
        if weight is None:
            continue
        total_weight += weight
        confidence = max(0, min(100, int(entry.get("confidence", 0) or 0)))
        if confidence >= threshold:
            confident_weight += weight
            confident_count += 1
    if total_weight == 0:
        return 0.0, 0
    return confident_weight / total_weight, confident_count


def _low_confidence_share(criteria: list[dict], threshold: int) -> float:
    if not criteria:
        return 1.0
    low_count = sum(
        1 for c in criteria
        if max(0, min(100, int(c.get("confidence", 0) or 0))) <= threshold
    )
    return low_count / len(criteria)


def evidence_coverage_is_too_low(criteria: list[dict], authenticity_score: int) -> tuple[bool, str]:
    if not criteria:
        return True, "No criteria were returned to score against."

    if all(c.get("score") is None for c in criteria):
        return True, "No criteria could be scored against the extracted evidence."

    if authenticity_score < LOW_COVERAGE_AUTHENTICITY_THRESHOLD:
        return True, f"Source authenticity was only {authenticity_score} percent."

    weighted_confidence = weighted_average_criteria_confidence(criteria)
    if weighted_confidence < LOW_COVERAGE_CRITERIA_CONFIDENCE_THRESHOLD:
        return True, f"Weighted criteria confidence was only {weighted_confidence:.0f} percent."

    plain_confidence = average_criteria_confidence(criteria)
    if plain_confidence < LOW_COVERAGE_PLAIN_AVERAGE_THRESHOLD:
        return True, f"Average criteria confidence was only {plain_confidence:.0f} percent."

    weak_criterion = _low_confidence_high_weight_criterion(criteria)
    if weak_criterion is not None:
        name = weak_criterion.get("name") or weak_criterion.get("id", "a high-weight criterion")
        confidence = weak_criterion.get("confidence", 0)
        return True, f"{name} had only {confidence} percent confidence."

    confident_weight_share, confident_count = _confident_weight_share(
        criteria, LOW_COVERAGE_CONFIDENT_CRITERION_THRESHOLD
    )
    if (
        confident_weight_share < LOW_COVERAGE_MIN_CONFIDENT_WEIGHT_SHARE
        or confident_count < LOW_COVERAGE_MIN_CONFIDENT_CRITERIA_COUNT
    ):
        return True, (
            f"Only {confident_count} of {len(criteria)} criteria (covering "
            f"{confident_weight_share * 100:.0f} percent of scoring weight) reached at least "
            f"{LOW_COVERAGE_CONFIDENT_CRITERION_THRESHOLD} percent confidence."
        )

    low_share = _low_confidence_share(criteria, LOW_COVERAGE_LOW_CONFIDENCE_CRITERION_THRESHOLD)
    if low_share > LOW_COVERAGE_MAX_LOW_CONFIDENCE_SHARE:
        low_count = round(low_share * len(criteria))
        return True, (
            f"{low_count} of {len(criteria)} criteria had {LOW_COVERAGE_LOW_CONFIDENCE_CRITERION_THRESHOLD} "
            f"percent confidence or below."
        )

    return False, ""


def research_confidence_label(criteria: list[dict], authenticity_score: int,
                               coverage_insufficient: bool) -> str:
    if coverage_insufficient:
        return "Insufficient"

    weighted_confidence = weighted_average_criteria_confidence(criteria)
    if (
        authenticity_score >= RESEARCH_CONFIDENCE_HIGH_AUTHENTICITY
        and weighted_confidence >= RESEARCH_CONFIDENCE_HIGH_WEIGHTED_CONF
    ):
        return "High"
    if (
        authenticity_score >= RESEARCH_CONFIDENCE_MEDIUM_AUTHENTICITY
        and weighted_confidence >= RESEARCH_CONFIDENCE_MEDIUM_WEIGHTED_CONF
    ):
        return "Medium"
    return "Low"


def _people_search_hits_from_sources(cleaned_sources: list[dict]) -> list[dict]:
    for source in cleaned_sources or []:
        if source.get("source_name") == "people_search" and source.get("status") == "FOUND":
            return source.get("people_hits") or []
    return []


def _decision_maker_name_key(name: str) -> str:
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


def _merge_verified_people_hits_into_extraction(extraction: dict, cleaned_sources: list[dict],
                                                 caller: str = "unknown") -> dict:
    hits = _people_search_hits_from_sources(cleaned_sources)
    if not hits:
        return extraction

    existing = extraction.get("decision_makers") or []
    existing_keys = {_decision_maker_name_key(person.get("name", "")) for person in existing}

    added_names = []
    for hit in hits:
        if hit.get("confidence") not in ("HIGH", "MEDIUM"):
            continue
        name = hit.get("name", "")
        key = _decision_maker_name_key(name)
        if not key or key in existing_keys:
            continue
        if _looks_like_former_role(hit.get("title", ""), "", hit.get("snippet", "")):
            continue
        existing.append({
            "name": name,
            "title": hit.get("title", ""),
            "public_facing_score": 60 if hit.get("confidence") == "HIGH" else 40,
            "tenure_status": "UNKNOWN",
            "tenure_evidence": "Verified via LinkedIn search snippet, not independently dated.",
            "is_india_specific": bool(hit.get("india_location_signal")),
            "source_excerpt": f"{hit.get('title', '')} — {hit.get('snippet', '')}"[:260],
            "source": "",
            "linkedin_url": hit.get("url", "") if is_literal_linkedin_profile_url_safe(hit.get("url", "")) else "",
        })
        existing_keys.add(key)
        added_names.append(name)

    if added_names:
        extraction["decision_makers"] = existing
        note = extraction.get("csr_head_note", "") or ""
        if not note.strip() or "no csr head" in note.lower() or "not identified" in note.lower():
            extraction["csr_head_note"] = (
                f"{added_names[0]} appears in a verified LinkedIn search result tied to CSR/"
                f"sustainability at the company; role currency beyond the snippet is unconfirmed."
            )
        logger.info(
            "_merge_verified_people_hits_into_extraction recovered dropped decision-makers "
            "caller=%s names=%s", caller, added_names,
        )
    return extraction


def is_literal_linkedin_profile_url_safe(url: str) -> bool:
    return bool(_LINKEDIN_PROFILE_URL.match((url or "").strip()))


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

    if isinstance(sanitized.get("decision_makers"), list) and isinstance(parsed.get("decision_makers"), list):
        raw_people = [p for p in parsed["decision_makers"] if isinstance(p, dict)]
        kept = []
        for sanitized_entry, raw_entry in zip(sanitized["decision_makers"], raw_people):
            matched, matched_text, matched_field = _looks_like_former_role_with_match(
                raw_entry.get("title", ""), raw_entry.get("tenure_evidence", ""),
                raw_entry.get("source_excerpt", ""),
            )
            if matched:
                logger.info(
                    "_repair_extraction dropped former-role decision maker caller=%s name=%r "
                    "matched_text=%r matched_field=%s",
                    caller, raw_entry.get("name", ""), matched_text, matched_field,
                )
                continue
            kept.append(sanitized_entry)
        sanitized["decision_makers"] = kept

    if isinstance(parsed.get("programmes"), list):
        raw_by_index = [p for p in parsed["programmes"] if isinstance(p, dict)]
        for sanitized_entry, raw_entry in zip(sanitized.get("programmes", []), raw_by_index):
            if isinstance(sanitized_entry, dict):
                sanitized_entry["chain_missing_elements"] = _sanitize_chain_missing_elements(
                    raw_entry.get("chain_missing_elements")
                )

    if isinstance(sanitized.get("spend"), dict):
        sanitized["spend"] = _apply_derived_spend_trend(sanitized["spend"])
    if isinstance(sanitized.get("eligibility"), dict):
        sanitized["eligibility"] = _apply_derived_profit_trend(sanitized["eligibility"])

    company_for_directives = str(parsed.get("_company_hint", "") or "")
    sanitized["key_facts_summary"] = str(parsed.get("key_facts_summary", "") or "")[:1500]
    sanitized["search_directives"] = _sanitize_search_directives(
        company_for_directives, parsed.get("search_directives"), cap=MAX_SEARCH_DIRECTIVES,
    )
    if not sanitized["search_directives"] and isinstance(parsed.get("open_questions"), list):
        sanitized["search_directives"] = _sanitize_search_directives(
            company_for_directives, parsed.get("open_questions"), cap=MAX_SEARCH_DIRECTIVES,
        )
    sanitized["open_questions"] = [
        d["question"] for d in sanitized["search_directives"] if d.get("question")
    ][:MAX_SEARCH_DIRECTIVES]
    return sanitized


def _empty_criteria() -> list[dict]:
    return [
        {
            "id": criterion_id, "name": CRITERIA_TITLES[criterion_id],
            "score": None, "confidence": 0,
            "evidence": "No signal returned for this criterion", "reasoning": "", "source": "",
        }
        for criterion_id in CRITERIA_IDS
    ]


def build_extraction_only_result(extraction: dict, mode: str) -> dict:
    merged = dict(extraction)
    merged.pop("key_facts_summary", None)
    merged["criteria"] = _empty_criteria()
    merged["fit_score"] = None
    merged["research_coverage"] = 0
    merged["fit_rationale"] = ""
    merged["overall_semantic_alignment"] = 0
    merged["alignment_rationale"] = ""
    merged["strategic_insight"] = LLM_SCORING_UNAVAILABLE_NOTE
    merged["scoring_incomplete"] = True
    merged["unscored_criteria_search_directives"] = []

    validated = _repair_analysis(merged)
    result = validated.model_dump()
    result["scoring_incomplete"] = True
    result["open_questions"] = [q.strip()[:200] for q in extraction.get("open_questions", []) if q and q.strip()][:MAX_SEARCH_DIRECTIVES]
    result["search_directives"] = extraction.get("search_directives", [])
    result["research_confidence_label"] = "Insufficient"

    logger.warning(
        "build_extraction_only_result company_facts_preserved mode=%r authenticity=%d "
        "partners=%d programmes=%d decision_makers=%d",
        mode, result["overall_authenticity_score"], len(result["partners"]),
        len(result["programmes"]), len(result["decision_makers"]),
    )
    return result


def _repair_analysis(parsed: dict) -> FullAnalysisSchema:
    parsed = dict(parsed) if isinstance(parsed, dict) else {}
    original_programmes = parsed.get("programmes") if isinstance(parsed.get("programmes"), list) else []
    original_decision_makers = parsed.get("decision_makers") if isinstance(parsed.get("decision_makers"), list) else []
    company_for_directives = str(parsed.get("_company_hint", "") or "")
    raw_unscored_directives = parsed.get("unscored_criteria_search_directives")
    parsed = _sanitize_dict_for_model(parsed, FullAnalysisSchema)
    parsed["unscored_criteria_search_directives"] = _sanitize_search_directives(
        company_for_directives, raw_unscored_directives, cap=MAX_SEARCH_DIRECTIVES,
        restrict_target_field_to=set(CRITERIA_IDS),
    )

    if isinstance(parsed.get("programmes"), list):
        raw_by_index = [p for p in original_programmes if isinstance(p, dict)]
        for sanitized_entry, raw_entry in zip(parsed["programmes"], raw_by_index):
            if isinstance(sanitized_entry, dict):
                sanitized_entry["chain_missing_elements"] = _sanitize_chain_missing_elements(
                    raw_entry.get("chain_missing_elements")
                )

    if isinstance(parsed.get("decision_makers"), list):
        raw_people = [p for p in original_decision_makers if isinstance(p, dict)]
        kept = []
        for sanitized_entry, raw_entry in zip(parsed["decision_makers"], raw_people):
            matched, matched_text, matched_field = _looks_like_former_role_with_match(
                raw_entry.get("title", ""), raw_entry.get("tenure_evidence", ""),
                raw_entry.get("source_excerpt", ""),
            )
            if matched:
                logger.info(
                    "_repair_analysis dropped former-role decision maker name=%r "
                    "matched_text=%r matched_field=%s",
                    raw_entry.get("name", ""), matched_text, matched_field,
                )
                continue
            kept.append(sanitized_entry)
        parsed["decision_makers"] = kept

    if isinstance(parsed.get("spend"), dict):
        parsed["spend"] = _apply_derived_spend_trend(parsed["spend"])
    if isinstance(parsed.get("eligibility"), dict):
        parsed["eligibility"] = _apply_derived_profit_trend(parsed["eligibility"])

    raw_criteria = parsed.get("criteria") if isinstance(parsed.get("criteria"), list) else []
    repaired_criteria, seen_ids = [], set()
    for entry in raw_criteria:
        if not isinstance(entry, dict):
            continue
        criterion_id = entry.get("id")
        if criterion_id not in CRITERIA_IDS or criterion_id in seen_ids:
            continue
        seen_ids.add(criterion_id)
        raw_score = entry.get("score", None)
        score_value = None
        if raw_score is not None:
            try:
                score_value = min(max(float(raw_score), 0), 5)
            except (TypeError, ValueError):
                score_value = None
        repaired_criteria.append({
            "id": criterion_id,
            "name": CRITERIA_TITLES[criterion_id],
            "score": score_value,
            "confidence": int(min(max(entry.get("confidence", 0) or 0, 0), 100)),
            "evidence": str(entry.get("evidence", ""))[:240],
            "reasoning": str(entry.get("reasoning", ""))[:240],
            "source": str(entry.get("source", "")),
        })
    for criterion_id in CRITERIA_IDS:
        if criterion_id not in seen_ids:
            repaired_criteria.append({
                "id": criterion_id, "name": CRITERIA_TITLES[criterion_id],
                "score": None, "confidence": 0,
                "evidence": "No signal returned for this criterion", "reasoning": "", "source": "",
            })
    ordered = {c["id"]: c for c in repaired_criteria}
    parsed["criteria"] = [ordered[cid] for cid in CRITERIA_IDS]
    parsed["research_coverage"] = compute_research_coverage(parsed["criteria"])

    null_scored_ids = {c["id"] for c in parsed["criteria"] if c.get("score") is None}
    parsed["unscored_criteria_search_directives"] = [
        d for d in parsed["unscored_criteria_search_directives"]
        if d.get("target_field") in null_scored_ids
    ]

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

    parsed = _enforce_decision_maker_narrative_consistency(parsed)

    try:
        return FullAnalysisSchema.model_validate(parsed)
    except ValidationError as exc:
        logger.warning("analysis validation failed, repairing containers error=%s", exc)
        for container_field, default in (
            ("spend", {}), ("entity_structure", {}), ("contact_pathway", {}), ("rfp_signal", {}),
            ("board_affinity", {}), ("volunteering", {}), ("group_foundation", {}),
            ("eligibility", {}), ("sector", {}),
            ("programmes", []), ("partners", []), ("decision_makers", []), ("geographies", []),
            ("red_flags", []), ("open_questions", []), ("search_directives", []),
            ("unscored_criteria_search_directives", []),
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
                "fit_score": (
                    int(min(max(parsed.get("fit_score", 0) or 0, 0), 100))
                    if parsed.get("fit_score") is not None else None
                ),
                "research_coverage": int(min(max(parsed.get("research_coverage", 0) or 0, 0), 100)),
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
                ("spend", SpendSchema), ("entity_structure", EntityStructureSchema),
                ("contact_pathway", ContactPathwaySchema),
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
                ("red_flags", RedFlagSchema), ("search_directives", SearchDirectiveSchema),
                ("unscored_criteria_search_directives", SearchDirectiveSchema),
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
                    fit_score=(
                        int(min(max(parsed.get("fit_score", 0) or 0, 0), 100))
                        if parsed.get("fit_score") is not None else None
                    ),
                    research_coverage=int(min(max(parsed.get("research_coverage", 0) or 0, 0), 100)),
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
    return max(0.0, _anthropic_cooldown_until - time.monotonic())


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

    prompt_blocks = _extraction_prompt_blocks(company, mission, evidence_text, sources_manifest)
    raw_reply = await call_anthropic_chat(
        prompt_blocks,
        temperature=0.0,
        max_tokens=EXTRACTION_OUTPUT_TOKEN_RESERVE,
        caller=f"extract_facts:{company}",
        use_prompt_caching=True,
    )
    if raw_reply is None:
        logger.error("extract_company_facts got no reply company=%r", company)
        return None

    parsed = parse_json_response(raw_reply, expected_keys=EXTRACTION_PRIORITY_KEYS, caller=f"extract_facts:{company}")
    if not parsed:
        logger.error("extract_company_facts empty parse company=%r", company)
        return None

    parsed["_company_hint"] = company
    extraction = _repair_extraction(parsed, caller=f"extract_facts:{company}")
    extraction = reconcile_extraction(extraction, working_sources)
    extraction = _merge_verified_people_hits_into_extraction(extraction, working_sources, caller=f"extract_facts:{company}")
    extraction = _reconcile_narrative_named_people_into_decision_makers(extraction, caller=f"extract_facts:{company}")
    extraction = _enforce_decision_maker_narrative_consistency(extraction)

    valid_sources = _valid_source_lookup(sources_manifest)
    extraction["delivery_model_source"] = _sanitize_source(extraction.get("delivery_model_source", ""), valid_sources)
    extraction.setdefault("spend", {})
    extraction["spend"]["source"] = _sanitize_source(extraction["spend"].get("source", ""), valid_sources)
    extraction["spend"]["trend_source"] = _sanitize_source(extraction["spend"].get("trend_source", ""), valid_sources)
    for entry in extraction["spend"].get("history", []) or []:
        entry["source"] = _sanitize_source(entry.get("source", ""), valid_sources)
    extraction.setdefault("entity_structure", {})
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
    for entry in extraction["eligibility"].get("net_profit_history", []) or []:
        entry["source"] = _sanitize_source(entry.get("source", ""), valid_sources)

    logger.info(
        "extract_company_facts DONE company=%r authenticity=%d partners=%d programmes=%d decision_makers=%d red_flags=%d "
        "spend_history_years=%d geographies=%d entity_structure=%r search_directives=%d",
        company, extraction.get("overall_authenticity_score", 0), len(extraction.get("partners", [])),
        len(extraction.get("programmes", [])), len(extraction.get("decision_makers", [])),
        len(extraction.get("red_flags", [])), len((extraction.get("spend") or {}).get("history", []) or []),
        len(extraction.get("geographies", [])), extraction.get("entity_structure", {}),
        len(extraction.get("search_directives", [])),
    )
    logger.info(
        "extract_company_facts FULL DUMP company=%r csr_head_note=%r delivery_model=%r sector=%r "
        "spend=%r eligibility=%r key_facts_summary=%r",
        company, extraction.get("csr_head_note", ""), extraction.get("delivery_model", ""),
        extraction.get("sector", {}), extraction.get("spend", {}), extraction.get("eligibility", {}),
        extraction.get("key_facts_summary", ""),
    )
    for programme in extraction.get("programmes", []) or []:
        logger.info(
            "extract_company_facts PROGRAMME company=%r name=%r what_is_funded=%r confidence=%r "
            "source_excerpt=%r",
            company, programme.get("name"), programme.get("what_is_funded"),
            programme.get("confidence"), programme.get("source_excerpt"),
        )
    for partner in extraction.get("partners", []) or []:
        logger.info(
            "extract_company_facts PARTNER company=%r name=%r relationship_type=%r confidence=%r "
            "source_excerpt=%r",
            company, partner.get("name"), partner.get("relationship_type"),
            partner.get("confidence"), partner.get("source_excerpt"),
        )
    for person in extraction.get("decision_makers", []) or []:
        logger.info(
            "extract_company_facts DECISION_MAKER company=%r name=%r title=%r is_india_specific=%r "
            "source_excerpt=%r",
            company, person.get("name"), person.get("title"),
            person.get("is_india_specific"), person.get("source_excerpt"),
        )
    for flag in extraction.get("red_flags", []) or []:
        logger.info(
            "extract_company_facts RED_FLAG company=%r flag=%r severity=%r explanation=%r",
            company, flag.get("flag"), flag.get("severity"), flag.get("explanation"),
        )
    for directive in extraction.get("search_directives", []) or []:
        logger.info(
            "extract_company_facts SEARCH_DIRECTIVE company=%r question=%r search_query=%r "
            "target_field=%r priority=%r",
            company, directive.get("question"), directive.get("search_query"),
            directive.get("target_field"), directive.get("priority"),
        )
    return extraction


async def score_extracted_facts(
    company: str,
    mission: str,
    mode: str,
    extraction: dict,
    sources_manifest: str,
    cfg: dict | None = None,
    csr_obligation: dict | None = None,
) -> dict | None:
    scoring_facts = {
        k: v for k, v in extraction.items()
        if k not in ("open_questions", "search_directives", "key_facts_summary",
                      "overall_authenticity_score", "evidence_recency",
                      "source_quality_assessment", "csr_head_note")
    }
    prompt_blocks = _scoring_prompt_blocks(company, mission, mode, scoring_facts, sources_manifest, csr_obligation=csr_obligation)
    prompt_text_for_estimate = "\n\n".join(b["text"] for b in prompt_blocks)
    prompt_tokens = estimate_tokens(prompt_text_for_estimate)
    output_ceiling = _anthropic_context_window() - SCORING_OUTPUT_TOKEN_RESERVE

    if prompt_tokens > output_ceiling:
        trimmed_facts = dict(scoring_facts)
        for list_field in ("programmes", "partners", "decision_makers", "geographies", "red_flags"):
            if trimmed_facts.get(list_field):
                trimmed_facts[list_field] = trimmed_facts[list_field][:5]
        prompt_blocks = _scoring_prompt_blocks(company, mission, mode, trimmed_facts, sources_manifest, csr_obligation=csr_obligation)
        prompt_text_for_estimate = "\n\n".join(b["text"] for b in prompt_blocks)
        prompt_tokens = estimate_tokens(prompt_text_for_estimate)

    if prompt_tokens > output_ceiling:
        logger.error(
            "score_extracted_facts could not fit prompt within context window company=%r prompt_tokens=%d ceiling=%d",
            company, prompt_tokens, output_ceiling,
        )
        return None

    raw_reply = await call_anthropic_chat(
        prompt_blocks,
        temperature=0.0,
        max_tokens=SCORING_OUTPUT_TOKEN_RESERVE,
        caller=f"score_facts:{company}",
        use_prompt_caching=True,
    )
    if raw_reply is None:
        logger.error("score_extracted_facts got no reply company=%r", company)
        return None

    parsed = parse_json_response(raw_reply, expected_keys=["criteria"], caller=f"score_facts:{company}")
    if not parsed:
        logger.error("score_extracted_facts empty parse company=%r", company)
        return None

    parsed["_company_hint"] = company
    parsed["unscored_criteria_search_directives"] = _sanitize_search_directives(
        company, parsed.get("unscored_criteria_search_directives"), cap=MAX_SEARCH_DIRECTIVES,
        restrict_target_field_to=set(CRITERIA_IDS),
    )

    logger.info(
        "score_extracted_facts DONE company=%r criteria_count=%d unscored_search_directives=%d",
        company, len(parsed.get("criteria", []) or []),
        len(parsed.get("unscored_criteria_search_directives", []) or []),
    )
    for criterion in parsed.get("criteria", []) or []:
        logger.info(
            "score_extracted_facts CRITERION company=%r id=%r score=%r confidence=%r evidence=%r",
            company, criterion.get("id"), criterion.get("score"),
            criterion.get("confidence"), criterion.get("evidence"),
        )
    for directive in parsed.get("unscored_criteria_search_directives", []) or []:
        logger.info(
            "score_extracted_facts UNSCORED_SEARCH_DIRECTIVE company=%r question=%r search_query=%r "
            "target_field=%r priority=%r",
            company, directive.get("question"), directive.get("search_query"),
            directive.get("target_field"), directive.get("priority"),
        )
    logger.info(
        "score_extracted_facts NARRATIVE company=%r fit_rationale=%r alignment_rationale=%r "
        "strategic_insight=%r",
        company, parsed.get("fit_rationale", ""), parsed.get("alignment_rationale", ""),
        parsed.get("strategic_insight", ""),
    )
    return parsed


async def analyze_and_score_company(
    company: str,
    mission: str,
    cleaned_sources: list[dict],
    sources_manifest: str,
    mode: str = "deep",
    cfg: dict | None = None,
    precomputed_extraction: dict | None = None,
) -> dict | None:
    if precomputed_extraction is not None:
        extraction = precomputed_extraction
        logger.info(
            "analyze_and_score_company REUSING precomputed extraction company=%r mode=%r — "
            "skipping a redundant extraction call",
            company, mode,
        )
    else:
        extraction = await extract_company_facts(company, mission, cleaned_sources, sources_manifest)
    if not extraction:
        return None

    csr_obligation_signal = compute_csr_obligation_signal(
        extraction.get("eligibility", {}), extraction.get("spend", {})
    )

    scoring = await score_extracted_facts(
        company, mission, mode, extraction, sources_manifest, cfg=cfg, csr_obligation=csr_obligation_signal,
    )
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
    merged["fit_score"] = None
    merged["research_coverage"] = 0
    merged["fit_rationale"] = scoring.get("fit_rationale", "")
    merged["overall_semantic_alignment"] = scoring.get("overall_semantic_alignment", 0)
    merged["alignment_rationale"] = scoring.get("alignment_rationale", "")
    merged["strategic_insight"] = scoring.get("strategic_insight", "")
    merged["unscored_criteria_search_directives"] = scoring.get("unscored_criteria_search_directives", [])
    merged["scoring_incomplete"] = False
    merged["_company_hint"] = company
    merged = _reconcile_narrative_named_people_into_decision_makers(merged, caller=f"score_facts:{company}")

    validated = _repair_analysis(merged)
    result = validated.model_dump()

    result["csr_obligation_signal"] = csr_obligation_signal

    coverage_insufficient, coverage_reason = evidence_coverage_is_too_low(
        result["criteria"], result["overall_authenticity_score"]
    )
    result["evidence_coverage_insufficient"] = coverage_insufficient
    result["evidence_coverage_reason"] = coverage_reason
    result["average_criteria_confidence_pct"] = round(
        average_criteria_confidence(result["criteria"]), 1
    )
    result["weighted_criteria_confidence_pct"] = round(
        weighted_average_criteria_confidence(result["criteria"]), 1
    )

    result["research_confidence_label"] = research_confidence_label(
        result["criteria"], result["overall_authenticity_score"], coverage_insufficient,
    )

    if coverage_insufficient:
        result["fit_score_display_mode"] = "insufficient_evidence"
        result["fit_score_label"] = "Insufficient evidence to score confidently"
    else:
        result["fit_score_display_mode"] = "scored"
        result["fit_score_label"] = ""

    result["fit_score"] = compute_final_fit_score(
        criteria=result["criteria"],
        authenticity_score=result["overall_authenticity_score"],
        mode=mode,
        model_reported_score=None,
        company=company,
    )
    result["research_coverage"] = compute_research_coverage(result["criteria"])

    if result["fit_score"] is None and not coverage_insufficient:
        coverage_insufficient = True
        result["evidence_coverage_insufficient"] = True
        result["evidence_coverage_reason"] = result["evidence_coverage_reason"] or (
            "No criteria could be scored against the extracted evidence."
        )
        result["fit_score_display_mode"] = "insufficient_evidence"
        result["fit_score_label"] = "Insufficient evidence to score confidently"
        result["research_confidence_label"] = "Insufficient"

    if not result.get("strategic_insight", "").strip():
        result["strategic_insight"] = result.get("fit_rationale", "") or LLM_UNAVAILABLE_EVIDENCE

    result["open_questions"] = [q.strip()[:200] for q in extraction.get("open_questions", []) if q and q.strip()][:MAX_SEARCH_DIRECTIVES]
    result["search_directives"] = extraction.get("search_directives", [])

    logger.info(
        "analyze_and_score_company DONE company=%r mode=%s final_fit_score=%s research_coverage=%d "
        "authenticity=%d avg_criteria_confidence=%.1f weighted_criteria_confidence=%.1f coverage_insufficient=%s "
        "coverage_reason=%r research_confidence=%s partners=%d programmes=%d decision_makers=%d "
        "unscored_search_directives=%d reused_extraction=%s",
        company, mode, result["fit_score"], result["research_coverage"], result["overall_authenticity_score"],
        result["average_criteria_confidence_pct"], result["weighted_criteria_confidence_pct"],
        result["evidence_coverage_insufficient"], result["evidence_coverage_reason"],
        result["research_confidence_label"],
        len(result["partners"]), len(result["programmes"]), len(result["decision_makers"]),
        len(result.get("unscored_criteria_search_directives", [])), precomputed_extraction is not None,
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