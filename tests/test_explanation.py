"""Tests for the explanation layer.

The load-bearing test is additivity. Shapley values must reconstruct the model
margin exactly, so any wiring mistake — wrong estimator, wrong preprocessing,
wrong output scale, wrong one-hot aggregation — shows up as a violation. It is
one assertion that covers the whole pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from credit_trust_lab.config import Config, ExplanationConfig
from credit_trust_lab.data.codebook import FEATURES, FeatureKind
from credit_trust_lab.data.splits import make_splits
from credit_trust_lab.explanation import (
    additivity_error,
    assess_faithfulness,
    assess_stability,
    baseline_row,
    explain,
    model_margin,
)
from credit_trust_lab.explanation.stability import expected_random_jaccard
from credit_trust_lab.models.registry import MODEL_REGISTRY, build_model

FAST = ExplanationConfig(n_background=40, n_stability_refits=3, top_k=5, random_state=99)


@pytest.fixture
def fitted(synthetic_frame, synthetic_target):
    """Return a builder for a fitted model with its splits and background."""

    def _build(name: str):
        config = Config()
        splits = make_splits(synthetic_frame, synthetic_target, config.split)
        model = build_model(name, config.model).fit(splits.X_train, splits.y_train)
        background = splits.X_train.sample(FAST.n_background, random_state=1)
        return model, splits, background

    return _build


# --------------------------------------------------------------------------
# Attributions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_attributions_reconstruct_the_model_margin(name, fitted):
    """Additivity: base value plus contributions must equal the model's own score."""
    model, splits, background = fitted(name)
    attributions = explain(model, splits.X_test, background, FAST)
    assert additivity_error(model, splits.X_test, attributions) < 1e-8


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_attributions_are_in_codebook_vocabulary(name, fitted):
    """One column per codebook feature, not per design-matrix column."""
    model, splits, background = fitted(name)
    attributions = explain(model, splits.X_test, background, FAST)
    assert attributions.values.shape == (len(splits.X_test), len(FEATURES))
    assert set(attributions.feature_names) == {f.name for f in FEATURES}
    assert not any("=" in name for name in attributions.feature_names)


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_one_hot_columns_are_summed_not_averaged(name, fitted):
    """Aggregation must preserve additivity, which averaging would destroy."""
    model, splits, background = fitted(name)
    attributions = explain(model, splits.X_test, background, FAST)
    reconstructed = attributions.base_value + attributions.values.sum(axis=1)
    np.testing.assert_allclose(reconstructed, model_margin(model, splits.X_test), atol=1e-8)


def test_mean_absolute_importance_is_sorted(fitted):
    model, splits, background = fitted("logistic_regression")
    importance = explain(model, splits.X_test, background, FAST).mean_absolute()
    assert list(importance) == sorted(importance, reverse=True)
    assert len(importance) == len(FEATURES)


def test_top_features_respects_k(fitted):
    model, splits, background = fitted("logistic_regression")
    attributions = explain(model, splits.X_test, background, FAST)
    assert len(attributions.top_features(0, 3)) == 3


def test_model_margin_is_log_odds_of_the_probability(fitted):
    """The scale claim, checked rather than asserted in a docstring."""
    model, splits, _ = fitted("logistic_regression")
    probability = model.predict_proba_bad(splits.X_test)
    expected = np.log(probability / (1.0 - probability))
    np.testing.assert_allclose(model_margin(model, splits.X_test), expected, atol=1e-8)


def test_unsupported_model_type_is_rejected(fitted):
    model, splits, background = fitted("logistic_regression")

    class Impostor:
        name = "impostor"

    with pytest.raises(TypeError, match="No exact SHAP explainer"):
        explain(Impostor(), splits.X_test, background, FAST)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Stability
# --------------------------------------------------------------------------


def test_expected_random_jaccard_matches_the_closed_form():
    """Two random 5-subsets of 20 share 1.25 elements, union 8.75."""
    assert expected_random_jaccard(5, 20) == pytest.approx(1.25 / 8.75)
    assert expected_random_jaccard(20, 20) == pytest.approx(1.0)


def test_stability_measures_lie_in_their_ranges(fitted):
    model, splits, background = fitted("logistic_regression")
    attributions = explain(model, splits.X_test, background, FAST)
    report = assess_stability(
        model.name, attributions, splits, splits.X_test, background, FAST, Config().model
    )
    assert 0.0 <= report.top_k_jaccard <= 1.0
    assert 0.0 <= report.top_feature_agreement <= 1.0
    assert 0.0 <= report.sign_agreement <= 1.0
    assert -1.0 <= report.global_rank_correlation <= 1.0
    assert report.n_refits == FAST.n_stability_refits


def test_stability_beats_chance(fitted):
    """A model fitted on real signal must reproduce more than a coin flip would."""
    model, splits, background = fitted("logistic_regression")
    attributions = explain(model, splits.X_test, background, FAST)
    report = assess_stability(
        model.name, attributions, splits, splits.X_test, background, FAST, Config().model
    )
    assert report.top_k_jaccard > report.random_jaccard
    assert report.top_feature_agreement > report.random_top_feature_agreement


def test_stability_reports_every_feature_spread(fitted):
    model, splits, background = fitted("logistic_regression")
    attributions = explain(model, splits.X_test, background, FAST)
    report = assess_stability(
        model.name, attributions, splits, splits.X_test, background, FAST, Config().model
    )
    assert set(report.per_feature_spread) == {f.name for f in FEATURES}
    spreads = list(report.per_feature_spread.values())
    assert spreads == sorted(spreads, reverse=True)


# --------------------------------------------------------------------------
# Faithfulness
# --------------------------------------------------------------------------


def test_baseline_uses_median_for_numeric_and_mode_for_coded(synthetic_frame):
    baseline = baseline_row(synthetic_frame)
    assert set(baseline.index) == {f.name for f in FEATURES}
    for feature in FEATURES:
        column = synthetic_frame[feature.name]
        if feature.kind is FeatureKind.NUMERIC:
            assert baseline[feature.name] == pytest.approx(float(column.median()))
        else:
            assert baseline[feature.name] in set(feature.codes)


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_named_features_move_the_model_more_than_random_ones(name, fitted):
    """The faithfulness claim itself: attribution must predict sensitivity."""
    model, splits, background = fitted(name)
    attributions = explain(model, splits.X_test, background, FAST)
    report = assess_faithfulness(model, attributions, splits.X_test, splits.X_train, FAST)
    assert report.effect_ratio > 1.0
    assert report.top_k_effect > report.random_k_effect


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_masking_moves_the_prediction_in_the_predicted_direction(name, fitted):
    """Removing a risk-increasing characteristic must lower predicted risk."""
    model, splits, background = fitted(name)
    attributions = explain(model, splits.X_test, background, FAST)
    report = assess_faithfulness(model, attributions, splits.X_test, splits.X_train, FAST)
    assert report.direction_agreement > 0.75


def test_deletion_curve_is_cumulative_and_rises(fitted):
    """Masking more of the named features cannot move the model less overall."""
    model, splits, background = fitted("logistic_regression")
    attributions = explain(model, splits.X_test, background, FAST)
    report = assess_faithfulness(model, attributions, splits.X_test, splits.X_train, FAST)
    assert len(report.deletion_curve) == FAST.top_k
    assert report.deletion_curve[-1] >= report.deletion_curve[0]


def test_faithfulness_report_serialises(fitted):
    model, splits, background = fitted("logistic_regression")
    attributions = explain(model, splits.X_test, background, FAST)
    payload = assess_faithfulness(
        model, attributions, splits.X_test, splits.X_train, FAST
    ).as_dict()
    assert "effect_ratio" in payload
    assert payload["n_cases"] == len(splits.X_test)
