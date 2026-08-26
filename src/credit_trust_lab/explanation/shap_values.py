"""SHAP attributions, aggregated back to the features a human recognises.

Two design decisions carry the weight here.

**Attributions are aggregated to codebook features, not design-matrix columns.**
The model sees ``purpose=3``, ``purpose=4``, …, ``housing=2``. A person reading
a reason code sees *purpose* and *housing*. Because Shapley values are additive,
the contribution of a one-hot group is exactly the sum of its columns'
attributions, so the aggregation is not an approximation — it is the correct
quantity, expressed in the vocabulary of the coding table.

**Attributions live on the log-odds scale.** Both explainers are asked for the
model's raw margin rather than its probability output. On the log-odds scale
the contributions sum exactly to the prediction minus the base value; on the
probability scale, the sigmoid destroys that additivity and the numbers stop
being Shapley values in any useful sense. The consequence for reading them: an
attribution of +0.8 means "this characteristic multiplied the odds of default
by about 2.2", not "+80 percentage points".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from credit_trust_lab.config import ExplanationConfig
from credit_trust_lab.data.codebook import FEATURES
from credit_trust_lab.data.preprocess import expected_feature_names
from credit_trust_lab.models.base import CreditModel
from credit_trust_lab.models.gradient_boosting import GradientBoostingScorer
from credit_trust_lab.models.logistic import LogisticScorecard


@dataclass(frozen=True)
class Attributions:
    """Per-case SHAP attributions in codebook vocabulary.

    Attributes:
        values: Array of shape ``(n_cases, n_features)`` on the log-odds scale.
        feature_names: Codebook feature names, matching ``values`` columns.
        base_value: The explainer's expected model output over the background
            distribution, on the log-odds scale.
        case_index: Row labels of the explained cases.
        model_name: Which model produced them.

    """

    values: np.ndarray
    feature_names: tuple[str, ...]
    base_value: float
    case_index: np.ndarray
    model_name: str

    def as_frame(self) -> pd.DataFrame:
        """Return attributions as a labelled frame."""
        return pd.DataFrame(self.values, columns=list(self.feature_names), index=self.case_index)

    def mean_absolute(self) -> pd.Series:
        """Global importance: mean |attribution| per feature, descending."""
        return self.as_frame().abs().mean().sort_values(ascending=False)

    def top_features(self, case: int, k: int) -> list[str]:
        """Return the ``k`` features with the largest absolute attribution for one case.

        Args:
            case: Positional index into ``values``.
            k: How many features to return.

        Returns:
            Feature names, most influential first.

        """
        order = np.argsort(-np.abs(self.values[case]))[:k]
        return [self.feature_names[i] for i in order]


def _group_columns_by_feature() -> dict[str, list[int]]:
    """Map each codebook feature to its design-matrix column positions."""
    design_columns = expected_feature_names()
    position = {name: i for i, name in enumerate(design_columns)}
    grouped: dict[str, list[int]] = {}
    for feature in FEATURES:
        if feature.name in position:  # scaled: one column, same name
            grouped[feature.name] = [position[feature.name]]
        else:  # encoded: one column per retained level, named "feature=code"
            grouped[feature.name] = [
                index for name, index in position.items() if name.startswith(f"{feature.name}=")
            ]
    return grouped


def _aggregate_to_features(design_values: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    """Sum design-matrix attributions into one column per codebook feature.

    Args:
        design_values: Array of shape ``(n_cases, n_design_columns)``.

    Returns:
        Aggregated values and the matching feature names.

    Raises:
        ValueError: If the input width does not match the codebook.

    """
    grouped = _group_columns_by_feature()
    expected_width = len(expected_feature_names())
    if design_values.shape[1] != expected_width:
        raise ValueError(
            f"SHAP returned {design_values.shape[1]} columns "
            f"but the design matrix has {expected_width}."
        )
    names = tuple(grouped)
    aggregated = np.column_stack([design_values[:, grouped[name]].sum(axis=1) for name in names])
    return aggregated, names


def _design_matrix(model: CreditModel, X: pd.DataFrame) -> np.ndarray:
    """Run the model's own fitted preprocessing, so explainer and model agree."""
    return np.asarray(model.pipeline.named_steps["preprocess"].transform(X), dtype=float)


def model_margin(model: CreditModel, X: pd.DataFrame) -> np.ndarray:
    """Return the model's raw additive score, before the sigmoid.

    The scale SHAP attributions live on, and therefore the scale every
    explanation check must use. On the probability scale the sigmoid compresses
    movement near 0 and 1, so identical model behaviour would register as a
    weaker response for confident cases.

    Args:
        model: A fitted model.
        X: Cases to score, in codebook form.

    Returns:
        Log-odds of P(bad credit), shape ``(n_samples,)``.

    """
    estimator = model.pipeline.named_steps["estimator"]
    return np.asarray(estimator.decision_function(_design_matrix(model, X)), dtype=float).ravel()


def explain(
    model: CreditModel,
    X_explain: pd.DataFrame,
    X_background: pd.DataFrame,
    config: ExplanationConfig | None = None,
) -> Attributions:
    """Compute SHAP attributions for a fitted model.

    Dispatches on model type because both baselines admit *exact* Shapley
    values — there is no reason to accept the sampling error of a model-
    agnostic explainer here, and doing so would confound explanation noise with
    the instability this module is built to measure.

    Args:
        model: A fitted base model.
        X_explain: Cases to explain, in codebook form.
        X_background: Rows defining the reference distribution, normally drawn
            from the training partition. Never from test.
        config: Sampling sizes and seed.

    Returns:
        Attributions aggregated to codebook features.

    Raises:
        TypeError: If the model type has no exact explainer registered.

    """
    config = config or ExplanationConfig()

    # Dispatch before touching the model, so an unsupported type fails with the
    # explanation this project owes the reader rather than an AttributeError.
    if not isinstance(model, LogisticScorecard | GradientBoostingScorer):
        raise TypeError(
            f"No exact SHAP explainer registered for {type(model).__name__}. "
            "Add one rather than falling back to a sampling explainer, whose "
            "noise would be indistinguishable from genuine instability."
        )

    background = _design_matrix(model, X_background)
    explain_matrix = _design_matrix(model, X_explain)
    estimator = model.pipeline.named_steps["estimator"]

    if isinstance(model, LogisticScorecard):
        explainer = shap.LinearExplainer(estimator, background)
        raw = np.asarray(explainer.shap_values(explain_matrix), dtype=float)
        base_value = float(np.asarray(explainer.expected_value).ravel()[0])
    else:
        explainer = shap.TreeExplainer(estimator)
        raw = np.asarray(explainer.shap_values(explain_matrix), dtype=float)
        base_value = float(np.asarray(explainer.expected_value).ravel()[0])

    if raw.ndim == 3:  # some versions return (n_cases, n_columns, n_classes)
        raw = raw[:, :, -1]

    values, names = _aggregate_to_features(raw)
    return Attributions(
        values=values,
        feature_names=names,
        base_value=base_value,
        case_index=np.asarray(X_explain.index),
        model_name=model.name,
    )


def additivity_error(
    model: CreditModel, X_explain: pd.DataFrame, attributions: Attributions
) -> float:
    """Largest violation of the SHAP additivity identity.

    Shapley values must satisfy ``base_value + sum(attributions) = model
    margin`` for every case. Checking it is the cheapest possible guard against
    an explainer being wired to the wrong model, the wrong preprocessing, or
    the wrong output scale — all of which produce plausible-looking numbers
    that mean nothing.

    Args:
        model: The fitted model the attributions came from.
        X_explain: The same cases that were explained.
        attributions: The attributions to verify.

    Returns:
        Maximum absolute discrepancy over the explained cases, on the log-odds
        scale.

    """
    margin = model_margin(model, X_explain)
    reconstructed = attributions.base_value + attributions.values.sum(axis=1)
    return float(np.max(np.abs(reconstructed - margin)))
