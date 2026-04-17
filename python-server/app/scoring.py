"""
RiskScoringEngine: orchestrates per-model inference and combines scores
using configurable weighted logic with confidence estimation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from app.schemas import FeatureSet, PredictResponse, RiskLevel
from app.core.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

# ── Default weights — must sum to 1.0 ─────────────────────────────────────────
DEFAULT_WEIGHTS: Dict[str, float] = {
    "sleep":         0.35,
    "typing":        0.25,
    "questionnaire": 0.40,
}

# ── Risk thresholds ────────────────────────────────────────────────────────────
RISK_THRESHOLDS = {
    RiskLevel.LOW:      (0.00, 0.30),
    RiskLevel.MODERATE: (0.30, 0.55),
    RiskLevel.HIGH:     (0.55, 0.75),
    RiskLevel.CRITICAL: (0.75, 1.01),
}


def _classify_risk(score: float) -> RiskLevel:
    for level, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= score < hi:
            return level
    return RiskLevel.CRITICAL


def _estimate_confidence(n_sources: int, scores: list[float]) -> float:
    """
    Confidence scales with:
      - number of data sources available (more = higher confidence)
      - agreement between scores (lower std = higher confidence)
    """
    source_factor = n_sources / 3.0
    if len(scores) > 1:
        agreement_factor = 1.0 - min(float(np.std(scores)), 0.5) / 0.5
    else:
        agreement_factor = 0.6   # single-source is inherently less certain

    return round(0.5 * source_factor + 0.5 * agreement_factor, 4)


class RiskScoringEngine:
    def __init__(self, registry: ModelRegistry, weights: Optional[Dict[str, float]] = None):
        self.registry = registry
        self.weights  = weights or DEFAULT_WEIGHTS
        # Normalize weights in case they don't sum to 1
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def score(self, features: FeatureSet) -> PredictResponse:
        raw_scores: Dict[str, float] = {}

        # ── Per-model inference ───────────────────────────────────────────────
        if features.sleep is not None:
            model = self.registry.get("sleep")
            if model:
                vec = np.array([features.sleep.to_vector()])
                raw_scores["sleep"] = model.predict_proba(vec)

        if features.typing is not None:
            model = self.registry.get("typing")
            if model:
                vec = np.array([features.typing.to_vector()])
                raw_scores["typing"] = model.predict_proba(vec)

        if features.questionnaire is not None:
            model = self.registry.get("questionnaire")
            if model:
                vec = np.array([features.questionnaire.to_vector()])
                raw_scores["questionnaire"] = model.predict_proba(vec)

        # ── Weighted combination (only over available sources) ────────────────
        available_weights = {k: self.weights[k] for k in raw_scores}
        weight_total = sum(available_weights.values())
        normalised_weights = {k: v / weight_total for k, v in available_weights.items()}

        combined = sum(
            raw_scores[src] * normalised_weights[src]
            for src in raw_scores
        )
        combined = round(float(np.clip(combined, 0.0, 1.0)), 4)

        active_scores = list(raw_scores.values())
        confidence    = _estimate_confidence(len(active_scores), active_scores)
        risk_level    = _classify_risk(combined)

        logger.info(
            f"Risk score computed: combined={combined:.4f} "
            f"level={risk_level} sources={list(raw_scores.keys())} "
            f"confidence={confidence:.4f}"
        )

        return PredictResponse(
            sleep_risk         = round(raw_scores.get("sleep"),         4) if "sleep"         in raw_scores else None,
            typing_anomaly     = round(raw_scores.get("typing"),        4) if "typing"        in raw_scores else None,
            questionnaire_risk = round(raw_scores.get("questionnaire"), 4) if "questionnaire" in raw_scores else None,
            combined_risk      = combined,
            risk_level         = risk_level,
            confidence         = confidence,
            model_version      = self.registry.version,
            weights_used       = normalised_weights,
        )