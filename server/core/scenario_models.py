from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class EvidenceKind(str, Enum):
    FACT = "FACT"
    ASSUMPTION = "ASSUMPTION"
    INFERENCE = "INFERENCE"
    SPECULATION = "SPECULATION"
    SOURCE = "SOURCE"
    UNCERTAINTY = "UNCERTAINTY"


class TimelinePhase(str, Enum):
    IMMEDIATE = "immediate"
    HOURS_DAYS = "hours_days"
    WEEKS_MONTHS = "weeks_months"
    YEARS = "years"
    DECADES_CENTURIES = "decades_centuries"


class CausalLevel(str, Enum):
    HYPOTHESIS = "HYPOTHESIS"
    FIRST_ORDER = "FIRST-ORDER EFFECT"
    SECOND_ORDER = "SECOND-ORDER EFFECT"
    THIRD_ORDER = "THIRD-ORDER EFFECT"
    LONG_TERM = "LONG-TERM CONSEQUENCE"


class SourceReference(BaseModel):
    title: str = Field(min_length=1)
    url: Optional[str] = None
    publisher: Optional[str] = None
    published_at: Optional[str] = None
    reference: Optional[str] = None

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.startswith(("http://", "https://")):
            raise ValueError("source URLs must use http:// or https://")
        return value

    def model_post_init(self, __context: Any) -> None:
        if not self.url and not self.reference:
            raise ValueError("sources must include a URL or provider reference")


class ResearchItem(BaseModel):
    kind: EvidenceKind
    claim: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    assumptions: List[str] = Field(default_factory=list)
    uncertainty: List[str] = Field(default_factory=list)
    source_ids: List[str] = Field(default_factory=list)


class ResearchDocument(BaseModel):
    items: List[ResearchItem] = Field(default_factory=list)
    sources: Dict[str, SourceReference] = Field(default_factory=dict)
    summary: str = ""

    @field_validator("items")
    @classmethod
    def validate_source_items(cls, items: List[ResearchItem]) -> List[ResearchItem]:
        for item in items:
            if item.kind == EvidenceKind.SOURCE and not item.source_ids:
                raise ValueError("SOURCE items must reference at least one source")
        return items


class AlternativeOutcome(BaseModel):
    outcome: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    assumptions: List[str] = Field(default_factory=list)


class CausalNode(BaseModel):
    id: str = Field(min_length=1)
    level: CausalLevel
    claim: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    assumptions: List[str] = Field(default_factory=list)
    uncertainty: List[str] = Field(default_factory=list)
    source_ids: List[str] = Field(default_factory=list)
    alternative_outcomes: List[AlternativeOutcome] = Field(default_factory=list)
    parent_ids: List[str] = Field(default_factory=list)


class AnalysisBranch(BaseModel):
    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    assumptions: List[str] = Field(default_factory=list)
    uncertainty: List[str] = Field(default_factory=list)
    nodes: List[CausalNode] = Field(default_factory=list)


class AnalysisDocument(BaseModel):
    hypothesis: str = Field(min_length=1)
    branches: List[AnalysisBranch] = Field(default_factory=list, max_length=8)

    @field_validator("branches")
    @classmethod
    def validate_branch_nodes(cls, branches: List[AnalysisBranch]) -> List[AnalysisBranch]:
        level_order = {level: index for index, level in enumerate(CausalLevel)}
        for branch in branches:
            levels = [node.level for node in branch.nodes]
            if levels and levels[0] != CausalLevel.HYPOTHESIS:
                raise ValueError("each causal branch must begin with HYPOTHESIS")
            if any(level_order[current] < level_order[previous] for previous, current in zip(levels, levels[1:])):
                raise ValueError("causal branch levels must move forward, never backward")
        return branches


class TimelineEntry(BaseModel):
    id: str = Field(min_length=1)
    phase: TimelinePhase
    title: str = Field(min_length=1)
    consequence: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    assumptions: List[str] = Field(default_factory=list)
    uncertainty: List[str] = Field(default_factory=list)
    alternative_outcomes: List[AlternativeOutcome] = Field(default_factory=list)
    causal_node_ids: List[str] = Field(default_factory=list)


class TimelineDocument(BaseModel):
    entries: List[TimelineEntry] = Field(default_factory=list)


class ScriptDocument(BaseModel):
    title: str = Field(min_length=1)
    sections: List[Dict[str, Any]] = Field(default_factory=list)
    narration: str = Field(min_length=1)


class VisualPlanDocument(BaseModel):
    scenes: List[Dict[str, Any]] = Field(default_factory=list)


class ScenarioCreateRequest(BaseModel):
    premise: str = Field(min_length=5, max_length=2000)
    title: Optional[str] = Field(default=None, max_length=200)


class ScenarioResponse(BaseModel):
    id: str
    premise: str
    title: str
    status: str
    created_at: Any
    updated_at: Any
    current_stage: Optional[str] = None
    job_id: Optional[str] = None
    error: Optional[str] = None
    research_data: Optional[Dict[str, Any]] = None
    analysis_data: Optional[Dict[str, Any]] = None
    timeline_data: Optional[Dict[str, Any]] = None
    script_data: Optional[Dict[str, Any]] = None
    visual_plan_data: Optional[Dict[str, Any]] = None
    artifact_dir: str
