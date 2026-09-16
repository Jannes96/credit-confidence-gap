"""Model registry, so that scripts iterate over models instead of naming them."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from credit_confidence_gap.config import ModelConfig
from credit_confidence_gap.models.base import CreditModel
from credit_confidence_gap.models.gradient_boosting import GradientBoostingScorer
from credit_confidence_gap.models.logistic import LogisticScorecard

MODEL_REGISTRY: Mapping[str, Callable[[ModelConfig | None], CreditModel]] = {
    LogisticScorecard.name: LogisticScorecard,
    GradientBoostingScorer.name: GradientBoostingScorer,
}


def build_model(name: str, config: ModelConfig | None = None) -> CreditModel:
    """Instantiate a registered model by name.

    Args:
        name: Registry key, e.g. ``"logistic_regression"``.
        config: Hyperparameters passed to the model.

    Returns:
        An unfitted model.

    Raises:
        KeyError: If ``name`` is not registered.

    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](config)


def build_all_models(config: ModelConfig | None = None) -> list[CreditModel]:
    """Instantiate every registered model.

    Args:
        config: Hyperparameters shared by all models.

    Returns:
        Unfitted models in registry order.

    """
    return [build_model(name, config) for name in MODEL_REGISTRY]
