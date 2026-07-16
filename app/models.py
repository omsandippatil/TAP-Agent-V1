from typing import Any, Literal

from pydantic import BaseModel


class ProspectRequest(BaseModel):
    company: str
    mode: Literal["screen", "deep"] = "screen"


class SourceRecord(BaseModel):
    source_name: str = ""
    priority: int = 0
    url: str = ""
    text: str = ""
    status: Literal["FOUND", "NOT_FOUND", "NOT_TRIED"] = "NOT_TRIED"
    fetch_method: str = ""
    cin: str = ""
    domain: str = ""
    people_hits: list[dict[str, Any]] = []
    plan_hits: list[dict[str, Any]] = []


class ScoringTier(BaseModel):
    min: int = 0
    tier: int = 0
    label: str = ""
    color: str = "#6B7280"
    key: str = ""
    action: str = ""
    description: str = ""


class ScoreBand(BaseModel):
    min: int = 0
    key: str = "LOW"
    label: str = ""
    color: str = "#9CA3A3"


class CriterionScore(BaseModel):
    id: str = ""
    name: str = ""
    score: float = 0
    confidence: int = 0
    evidence: str = ""
    reasoning: str = ""
    source: str = ""


class SpendYear(BaseModel):
    fiscal_year: str = ""
    inr_crore: float | None = None
    display: str = ""
    source: str = ""
    source_excerpt: str = ""


class SpendDetail(BaseModel):
    inr_crore: float | None = None
    display: str = ""
    fiscal_year: str = ""
    confidence: int = 0
    source_excerpt: str = ""
    source: str = ""
    trend_direction: str = "UNKNOWN"
    trend_evidence: str = ""
    trend_source: str = ""
    history: list[SpendYear] = []


class ProgrammeDetail(BaseModel):
    name: str = ""
    description: str = ""
    is_multi_year: bool = False
    cohort_or_scale: str = ""
    source_excerpt: str = ""
    source: str = ""


class PartnerDetail(BaseModel):
    name: str = ""
    relationship_type: str = ""
    is_known_tap_peer: bool = False
    similarity_to_tap: int = 0
    source_excerpt: str = ""
    source: str = ""


class GeographyDetail(BaseModel):
    place: str = ""
    is_tap_state: bool = False
    source_excerpt: str = ""
    source: str = ""


class RedFlagDetail(BaseModel):
    flag: str = ""
    severity: str = ""
    explanation: str = ""
    source: str = ""


class ContactPathwayDetail(BaseModel):
    channel: str = ""
    evidence: str = ""
    source: str = ""


class RfpSignalDetail(BaseModel):
    present: bool = False
    channel: str = ""
    evidence: str = ""
    source: str = ""


class BoardAffinityDetail(BaseModel):
    present: bool = False
    person_name: str = ""
    connection: str = ""
    source_excerpt: str = ""
    source: str = ""


class VolunteeringDetail(BaseModel):
    present: bool = False
    programme_name: str = ""
    description: str = ""
    source_excerpt: str = ""
    source: str = ""


class GroupFoundationDetail(BaseModel):
    routed_through_group: bool = False
    foundation_name: str = ""
    explanation: str = ""
    source_excerpt: str = ""
    source: str = ""


class EligibilityDetail(BaseModel):
    plausibly_mandated: str = "UNKNOWN"
    reasoning: str = ""
    net_worth_turnover_signal: str = ""
    source: str = ""


class SectorDetail(BaseModel):
    sector: str = "UNKNOWN"
    sub_sector: str = ""
    education_csr_prior: str = "MEDIUM"
    reasoning: str = ""


class DecisionMakerDetail(BaseModel):
    name: str = ""
    title: str = ""
    public_facing_score: int = 0
    tenure_status: str = "UNKNOWN"
    tenure_evidence: str = ""
    source_excerpt: str = ""
    source: str = ""


class CompanyAnalysis(BaseModel):
    overall_semantic_alignment: int = 0
    alignment_rationale: str = ""
    delivery_model: str = "UNCLEAR"
    delivery_model_evidence: str = ""
    delivery_model_source: str = ""
    spend: SpendDetail = SpendDetail()
    programmes: list[ProgrammeDetail] = []
    partners: list[PartnerDetail] = []
    decision_makers: list[DecisionMakerDetail] = []
    geographies: list[GeographyDetail] = []
    criteria: list[CriterionScore] = []
    red_flags: list[RedFlagDetail] = []
    contact_pathway: ContactPathwayDetail = ContactPathwayDetail()
    rfp_signal: RfpSignalDetail = RfpSignalDetail()
    board_affinity: BoardAffinityDetail = BoardAffinityDetail()
    volunteering: VolunteeringDetail = VolunteeringDetail()
    group_foundation: GroupFoundationDetail = GroupFoundationDetail()
    eligibility: EligibilityDetail = EligibilityDetail()
    sector: SectorDetail = SectorDetail()
    evidence_recency: str = ""
    csr_head_note: str = ""
    source_quality_assessment: str = ""
    overall_authenticity_score: int = 0
    open_questions: list[str] = []


class DecisionMaker(BaseModel):
    name: str = ""
    title: str = ""
    is_current_csr_role: bool = False
    match_confidence: int = 0
    linkedin_url: str = ""
    tenure_status: str = "UNKNOWN"
    reasoning: str = ""


class ImportantLink(BaseModel):
    label: str = ""
    url: str = ""
    relevance: str = ""


class ScreeningResult(BaseModel):
    state: Literal["FOUND", "NOT_FOUND_IN_SOURCE", "CONFIRMED_ABSENT"] = "NOT_FOUND_IN_SOURCE"
    fit_score: int = 0
    strategic_insight: str = ""
    band: ScoreBand = ScoreBand()
    scoring_tier: ScoringTier = ScoringTier()
    analysis: CompanyAnalysis | None = None
    score_breakdown: dict[str, Any] = {}
    decision_makers: list[DecisionMaker] = []
    sources: list[SourceRecord] = []
    source_links: list[dict[str, Any]] = []
    important_links: list[ImportantLink] = []
    cache_key: tuple[str, str, str] | None = None


class ApiHealthStatus(BaseModel):
    ok: bool = False
    model: str | None = None
    message: str = ""


class ApiHealthResponse(BaseModel):
    groq: ApiHealthStatus = ApiHealthStatus()
    google_search: dict[str, Any] = {}