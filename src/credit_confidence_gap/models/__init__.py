"""Scoring models behind one common interface."""

from credit_confidence_gap.models.base import CreditModel
from credit_confidence_gap.models.gradient_boosting import GradientBoostingScorer
from credit_confidence_gap.models.logistic import LogisticScorecard
from credit_confidence_gap.models.registry import MODEL_REGISTRY, build_all_models, build_model

__all__ = [
    "MODEL_REGISTRY",
    "CreditModel",
    "GradientBoostingScorer",
    "LogisticScorecard",
    "build_all_models",
    "build_model",
]
