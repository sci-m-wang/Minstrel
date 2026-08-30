from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CharacterSpec(Model):
    character_id: str
    character_name: str
    work: str
    anonymous_id: str
    panel: str
    aliases: list[str] = Field(default_factory=list)
    language: str = "en"
    iconicity: Literal["iconic", "medium", "subtle", "unknown"] = "unknown"
    gold_profile: str = ""

    @field_validator("character_id", "anonymous_id")
    @classmethod
    def nonempty_identifier(cls, value: str) -> str:
        if not value or any(ch.isspace() for ch in value):
            raise ValueError("identifier must be non-empty and contain no whitespace")
        return value

    @model_validator(mode="after")
    def include_canonical_alias(self) -> "CharacterSpec":
        if self.character_name not in self.aliases:
            self.aliases.insert(0, self.character_name)
        return self


class Comment(Model):
    comment_id: str
    character_id: str
    character_name: str
    work: str
    platform: str
    thread_id: str = ""
    author_hash: str
    timestamp: str = ""
    raw_text: str
    language: str
    source_url: str = ""
    collection_method: str = "authorized_export"
    license_note: str = ""
    is_synthetic: bool = False
    collected_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @field_validator("raw_text")
    @classmethod
    def meaningful_text(cls, value: str) -> str:
        value = " ".join(value.split())
        if len(value) < 12:
            raise ValueError("raw_text must contain at least 12 characters")
        return value


class Probe(Model):
    probe_id: str
    domain_id: str
    domain: str
    question_en: str
    question_zh: str
    search_terms: list[str] = Field(default_factory=list)

    def query(self, language: str = "en") -> str:
        question = self.question_zh if language.startswith("zh") else self.question_en
        return f"{question} {' '.join(self.search_terms)}".strip()


class RetrievedComment(Model):
    comment_id: str
    anonymous_id: str
    text: str
    platform: str
    author_hash: str
    language: str
    score: float
    rank_sources: list[str] = Field(default_factory=list)


CueStatus = Literal["SUPPORTED", "WEAK", "CONTESTED", "UNKNOWN"]


class Cue(Model):
    cue_id: str
    probe_id: str
    domain_id: str
    cue: str
    context: str
    cue_type: Literal[
        "behavioral_pattern",
        "motive",
        "value_priority",
        "appraisal_style",
        "affect_coping",
        "interpersonal_pattern",
        "narrative_theme",
        "expressive_signature",
        "unknown",
    ] = "behavioral_pattern"
    directness: Literal["behavior_based", "explicit_judgment", "inferred", "unknown"] = (
        "behavior_based"
    )
    support: list[str] = Field(default_factory=list)
    counterevidence: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    status: CueStatus = "UNKNOWN"


PERSON_MODEL_SECTIONS = [
    "stable_tendencies",
    "motives_and_goals",
    "value_priorities",
    "cognitive_appraisal_model",
    "affective_dynamics",
    "interpersonal_patterns",
    "self_narrative_themes",
    "situation_behavior_signatures",
    "expressive_signature",
    "contradictions_boundary_conditions",
    "unknown_contested",
]


class PersonModel(Model):
    anonymous_id: str
    sections: dict[str, list[str]]
    cue_ids: list[str] = Field(default_factory=list)
    model_name: str = ""
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @model_validator(mode="after")
    def require_all_sections(self) -> "PersonModel":
        missing = [name for name in PERSON_MODEL_SECTIONS if name not in self.sections]
        if missing:
            raise ValueError(f"missing person-model sections: {', '.join(missing)}")
        self.sections = {
            key: [str(item).strip() for item in self.sections[key] if str(item).strip()]
            for key in PERSON_MODEL_SECTIONS
        }
        return self


class BenchmarkExample(Model):
    example_id: str
    character_id: str
    query: str
    context: str = ""
    reference: str = ""
    benchmark: str = "custom"
    metadata: dict = Field(default_factory=dict)


class GenerationRecord(Model):
    run_id: str
    example_id: str
    character_id: str
    anonymous_id: str
    condition: str
    replicate: int
    query: str
    context: str = ""
    output: str
    actor_model: str
    score: float | None = None
    judge_rationale: str = ""
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
