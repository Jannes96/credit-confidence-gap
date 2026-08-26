"""Tests for feature preparation and the shared model interface."""

from __future__ import annotations

import numpy as np
import pytest

from credit_trust_lab.config import Config, ModelConfig
from credit_trust_lab.data.codebook import FEATURES_BY_NAME, FeatureKind, feature_names
from credit_trust_lab.data.preprocess import (
    build_preprocessor,
    describe_design_matrix,
    expected_feature_names,
    validate_frame,
)
from credit_trust_lab.data.splits import make_splits
from credit_trust_lab.evaluation.report import evaluate_model
from credit_trust_lab.models.base import NotFittedError
from credit_trust_lab.models.registry import MODEL_REGISTRY, build_all_models, build_model

# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------


def test_design_matrix_width_matches_the_codebook(synthetic_frame):
    transformed = build_preprocessor().fit_transform(synthetic_frame)
    assert transformed.shape[1] == len(expected_feature_names())


def test_expected_width_is_derivable_by_hand():
    """Scaled columns plus (levels - 1) per categorical, given drop='first'."""
    scaled = len(feature_names(FeatureKind.NUMERIC)) + len(feature_names(FeatureKind.ORDINAL))
    encoded = sum(
        len(FEATURES_BY_NAME[name].levels) - 1
        for name in feature_names(FeatureKind.NOMINAL) + feature_names(FeatureKind.BINARY)
    )
    assert len(expected_feature_names()) == scaled + encoded


def test_declared_categories_survive_absence_from_the_training_rows(synthetic_frame):
    """A level missing from the fitting rows must still get its column.

    This is why categories come from the codebook rather than from the data:
    otherwise a rare level absent from one split silently changes the design
    matrix between partitions.
    """
    without_business = synthetic_frame[synthetic_frame["purpose"] != 10]
    preprocessor = build_preprocessor().fit(without_business)
    assert preprocessor.transform(without_business).shape[1] == len(expected_feature_names())
    # And a row carrying the unseen level still transforms.
    with_business = synthetic_frame[synthetic_frame["purpose"] == 10]
    assert preprocessor.transform(with_business).shape[1] == len(expected_feature_names())


def test_transform_produces_no_missing_values(synthetic_frame):
    transformed = build_preprocessor().fit_transform(synthetic_frame)
    assert not np.isnan(transformed).any()


def test_scaled_columns_are_standardised(synthetic_frame):
    transformed = describe_design_matrix(build_preprocessor().fit_transform(synthetic_frame))
    assert transformed["duration"].mean() == pytest.approx(0.0, abs=1e-9)
    assert transformed["duration"].std(ddof=0) == pytest.approx(1.0, abs=1e-9)


def test_one_hot_columns_are_indicators(synthetic_frame):
    transformed = describe_design_matrix(build_preprocessor().fit_transform(synthetic_frame))
    assert set(np.unique(transformed["housing=2"].to_numpy())) <= {0.0, 1.0}


def test_ordinal_columns_are_not_one_hot_encoded(synthetic_frame):
    """The answer to the encoding question, pinned as a test."""
    names = expected_feature_names()
    assert "savings" in names
    assert not any(name.startswith("savings=") for name in names)


def test_validate_frame_rejects_missing_and_extra_columns(synthetic_frame):
    with pytest.raises(ValueError, match="missing columns"):
        validate_frame(synthetic_frame.drop(columns=["age"]))
    with pytest.raises(ValueError, match="undeclared columns"):
        validate_frame(synthetic_frame.assign(extra=1))


def test_validate_frame_rejects_missing_values(synthetic_frame):
    corrupted = synthetic_frame.copy().astype(float)
    corrupted.loc[0, "age"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        validate_frame(corrupted)


def test_describe_design_matrix_rejects_the_wrong_width():
    with pytest.raises(ValueError, match="columns"):
        describe_design_matrix(np.zeros((3, 2)))


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_every_model_satisfies_the_common_interface(name, synthetic_frame, synthetic_target):
    model = build_model(name)
    assert not model.is_fitted
    with pytest.raises(NotFittedError):
        model.predict_proba_bad(synthetic_frame)

    model.fit(synthetic_frame, synthetic_target)
    probabilities = model.predict_proba_bad(synthetic_frame)

    assert model.is_fitted
    assert probabilities.shape == (len(synthetic_frame),)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_probabilities_are_oriented_towards_bad_credit(name, synthetic_frame, synthetic_target):
    """P(bad) must correlate positively with the observed bad outcomes.

    A silently inverted probability column would still produce plausible
    metrics with the signs flipped; this is the guard against that.
    """
    model = build_model(name).fit(synthetic_frame, synthetic_target)
    probabilities = model.predict_proba_bad(synthetic_frame)
    bad = np.asarray(synthetic_target) == 1
    assert probabilities[bad].mean() > probabilities[~bad].mean()


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_models_are_deterministic_under_a_fixed_seed(name, synthetic_frame, synthetic_target):
    config = ModelConfig(random_state=99)
    first = build_model(name, config).fit(synthetic_frame, synthetic_target)
    second = build_model(name, config).fit(synthetic_frame, synthetic_target)
    np.testing.assert_allclose(
        first.predict_proba_bad(synthetic_frame), second.predict_proba_bad(synthetic_frame)
    )


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_decide_requires_an_explicit_threshold(name, synthetic_frame, synthetic_target):
    """There must be no ``predict`` that quietly reinstates 0.5."""
    model = build_model(name).fit(synthetic_frame, synthetic_target)
    assert not hasattr(model, "predict")
    strict = model.decide(synthetic_frame, threshold=0.9)
    lenient = model.decide(synthetic_frame, threshold=0.1)
    assert strict.sum() <= lenient.sum()


def test_unknown_model_name_is_rejected():
    with pytest.raises(KeyError, match="Unknown model"):
        build_model("deep_neural_scorecard")


def test_registry_builds_both_baselines():
    models = build_all_models()
    assert {m.name for m in models} == {"logistic_regression", "gradient_boosting"}


# --------------------------------------------------------------------------
# End-to-end wiring
# --------------------------------------------------------------------------


def test_threshold_is_fitted_on_calibration_and_applied_to_test(synthetic_frame, synthetic_target):
    """The evaluation record must not choose its threshold on the test data."""
    config = Config()
    splits = make_splits(synthetic_frame, synthetic_target, config.split)
    evaluation = evaluate_model(build_model("logistic_regression", config.model), splits, config)

    from credit_trust_lab.evaluation.costs import optimise_threshold

    on_calibration = optimise_threshold(
        np.asarray(splits.y_calibration),
        evaluation.calibration_partition.y_prob,
        config.cost,
    )
    assert evaluation.threshold.threshold == pytest.approx(on_calibration.threshold)


def test_evaluation_record_serialises(synthetic_frame, synthetic_target):
    import json

    config = Config()
    splits = make_splits(synthetic_frame, synthetic_target, config.split)
    evaluation = evaluate_model(build_model("logistic_regression", config.model), splits, config)
    payload = json.dumps(evaluation.as_dict(), default=float)
    assert "expected_calibration_error" in payload
    assert "theoretical_threshold" in payload
