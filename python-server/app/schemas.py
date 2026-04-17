"""
Pydantic v2 schemas for ML service request/response validation.
No PII — only structured numerical features.
"""
from __future__ import annotations
from typing import Optional, List
from enum import Enum

from pydantic import BaseModel, Field, model_validator


# ── Risk Level Enum ────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW      = "low"
    MODERATE = "moderate"
    HIGH     = "high"
    CRITICAL = "critical"


# ── Feature Schemas ────────────────────────────────────────────────────────────

class SleepFeatures(BaseModel):
    duration_hours:      float = Field(..., ge=0, le=24)
    efficiency_percent:  float = Field(..., ge=0, le=100)
    interruptions_count: int   = Field(..., ge=0)
    onset_latency_min:   float = Field(..., ge=0)
    deep_sleep_percent:  float = Field(..., ge=0, le=100)

    def to_vector(self) -> List[float]:
        return [
            self.duration_hours,
            self.efficiency_percent / 100.0,
            self.interruptions_count,
            self.onset_latency_min,
            self.deep_sleep_percent / 100.0,
        ]


class TypingFeatures(BaseModel):
    avg_wpm:              float = Field(..., ge=0)
    error_rate:           float = Field(..., ge=0, le=1)
    pause_frequency:      float = Field(..., ge=0)
    session_duration_min: float = Field(..., ge=0)
    backspace_rate:       float = Field(..., ge=0, le=1)

    def to_vector(self) -> List[float]:
        return [
            self.avg_wpm / 100.0,      # normalise
            self.error_rate,
            self.pause_frequency,
            self.session_duration_min / 60.0,
            self.backspace_rate,
        ]


class QuestionnaireFeatures(BaseModel):
    phq9_score:   int        = Field(..., ge=0, le=27)
    gad7_score:   int        = Field(..., ge=0, le=21)
    psqi_score:   int        = Field(..., ge=0, le=21)
    custom_items: List[float] = Field(default_factory=list, max_length=20)

    def to_vector(self) -> List[float]:
        base = [
            self.phq9_score / 27.0,
            self.gad7_score / 21.0,
            self.psqi_score / 21.0,
        ]
        return base + [v / 10.0 for v in self.custom_items[:10]]


class FeatureSet(BaseModel):
    sleep:         Optional[SleepFeatures]         = None
    typing:        Optional[TypingFeatures]        = None
    questionnaire: Optional[QuestionnaireFeatures] = None

    @model_validator(mode="after")
    def at_least_one_source(self) -> "FeatureSet":
        if not any([self.sleep, self.typing, self.questionnaire]):
            raise ValueError("At least one feature source (sleep/typing/questionnaire) is required")
        return self


# ── Request / Response ─────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    features:   FeatureSet
    request_id: Optional[str] = None


class ModelScores(BaseModel):
    sleep_risk:         Optional[float] = Field(None, ge=0, le=1)
    typing_anomaly:     Optional[float] = Field(None, ge=0, le=1)
    questionnaire_risk: Optional[float] = Field(None, ge=0, le=1)


class PredictResponse(BaseModel):
    sleep_risk:         Optional[float] = Field(None, ge=0, le=1)
    typing_anomaly:     Optional[float] = Field(None, ge=0, le=1)
    questionnaire_risk: Optional[float] = Field(None, ge=0, le=1)
    combined_risk:      float           = Field(..., ge=0, le=1)
    risk_level:         RiskLevel
    confidence:         float           = Field(..., ge=0, le=1)
    model_version:      str
    weights_used:       dict