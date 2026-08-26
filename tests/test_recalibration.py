"""Tests for post-hoc recalibration.

Two properties carry the most weight here. First, that no map ever inverts two
applicants, which is what licenses the claim that recalibration changes honesty
rather than ranking. That is deliberately weaker than ROC-AUC invariance:
isotonic regression is monotone but not injective, so it merges applicants into
ties and does move ROC-AUC — a cost the calibration metrics never show. Second,
that cross-fitting actually withholds each row's outcome from the calibrator
that scores it, which is what keeps the decision threshold honest.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from credit_trust_lab.config import Config, RecalibrationConfig
from credit_trust_lab.data.splits import make_splits
from credit_trust_lab.evaluation.calibration import expected_calibration_error
from credit_trust_lab.models.base import ScoringModel
from credit_trust_lab.models.registry import build_model
from credit_trust_lab.recalibration import (
    CALIBRATOR_REGISTRY,
    CalibratedModel,
    IdentityCalibrator,
    PlattCalibrator,
    build_calibrator,
    cross_fitted_probabilities,
    fit_all_calibrations,
    fit_calibration,
)
from credit_trust_lab.recalibration.base import NotFittedError, logit

# --------------------------------------------------------------------------
# The maps themselves
# --------------------------------------------------------------------------


@pytest.fixture
def overconfident(rng):
    """Stated probabilities pushed towards the extremes relative to the truth."""
    truth = rng.uniform(0.15, 0.85, size=2000)
    stated = np.clip(1.0 / (1.0 + np.exp(-2.5 * logit(truth))), 1e-4, 1 - 1e-4)
    return rng.binomial(1, truth), stated


@pytest.mark.parametrize("name", sorted(CALIBRATOR_REGISTRY))
def test_every_calibrator_satisfies_the_interface(name, overconfident):
    y, p = overconfident
    calibrator = build_calibrator(name)
    assert not calibrator.is_fitted
    with pytest.raises(NotFittedError):
        calibrator.transform(p)

    calibrator.fit(p, y)
    corrected = calibrator.transform(p)
    assert calibrator.is_fitted
    assert corrected.shape == p.shape
    assert np.all((corrected >= 0) & (corrected <= 1))


@pytest.mark.parametrize("name", sorted(CALIBRATOR_REGISTRY))
def test_no_map_ever_inverts_two_applicants(name, overconfident):
    """The property that makes recalibration a statement about honesty only.

    Monotonicity means the order is never reversed: if one applicant scored
    lower than another before, they cannot score higher afterwards. This is the
    correct statement, and it is weaker than "ROC-AUC is unchanged" — see
    ``test_isotonic_creates_ties_and_moves_auc``.
    """
    y, p = overconfident
    corrected = build_calibrator(name).fit(p, y).transform(p)
    order = np.argsort(p)
    assert np.all(np.diff(corrected[order]) >= -1e-12)


@pytest.mark.parametrize("name", ["identity", "platt"])
def test_strictly_monotone_maps_preserve_roc_auc_exactly(name, overconfident):
    y, p = overconfident
    corrected = build_calibrator(name).fit(p, y).transform(p)
    assert roc_auc_score(y, corrected) == pytest.approx(roc_auc_score(y, p), abs=1e-9)


def test_isotonic_creates_ties_and_moves_auc(overconfident):
    """Isotonic regression is monotone but not injective, and that costs ranking.

    Being a step function, it maps distinct scores onto identical values.
    Applicants merged onto one level can no longer be ranked against each
    other, so ROC-AUC moves — a real loss that no calibration metric reports,
    and the reason ``FittedCalibration.n_output_levels`` exists.
    """
    y, p = overconfident
    corrected = build_calibrator("isotonic").fit(p, y).transform(p)
    assert len(np.unique(corrected)) < len(np.unique(p)) / 10
    assert roc_auc_score(y, corrected) != pytest.approx(roc_auc_score(y, p), abs=1e-9)


def test_identity_returns_its_input_exactly(overconfident):
    y, p = overconfident
    np.testing.assert_array_equal(IdentityCalibrator().fit(p, y).transform(p), p)


def test_platt_leaves_already_calibrated_probabilities_alone(rng):
    """Fitting on the log-odds makes the identity reachable at a=1, b=0."""
    p = rng.uniform(0.05, 0.95, size=5000)
    y = rng.binomial(1, p)
    slope, intercept = PlattCalibrator().fit(p, y).coefficients
    assert slope == pytest.approx(1.0, abs=0.15)
    assert intercept == pytest.approx(0.0, abs=0.15)


def test_platt_pulls_in_overconfident_probabilities(overconfident):
    """A slope below 1 is the signature of shrinking extreme probabilities."""
    y, p = overconfident
    slope, _ = PlattCalibrator().fit(p, y).coefficients
    assert slope < 1.0
    assert expected_calibration_error(y, PlattCalibrator().fit(p, y).transform(p)) < (
        expected_calibration_error(y, p)
    )


def test_isotonic_output_is_non_decreasing(overconfident):
    y, p = overconfident
    corrected = build_calibrator("isotonic").fit(p, y).transform(p)
    order = np.argsort(p)
    assert np.all(np.diff(corrected[order]) >= -1e-12)


def test_calibrators_reject_malformed_input():
    calibrator = IdentityCalibrator()
    with pytest.raises(ValueError, match="Shape mismatch"):
        calibrator.fit(np.array([0.1, 0.2]), np.array([1]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        calibrator.fit(np.array([0.1, 1.7]), np.array([0, 1]))
    with pytest.raises(ValueError, match="only 0 and 1"):
        calibrator.fit(np.array([0.1, 0.2]), np.array([0, 5]))
    with pytest.raises(ValueError, match="empty"):
        calibrator.fit(np.array([]), np.array([]))


def test_unknown_calibrator_is_rejected():
    with pytest.raises(KeyError, match="Unknown calibrator"):
        build_calibrator("beta_calibration")


def test_logit_clips_away_from_the_asymptotes():
    """Gradient boosting produces exact 0 and 1; the transform must survive them."""
    assert np.isfinite(logit(np.array([0.0, 1.0]))).all()


# --------------------------------------------------------------------------
# Cross-fitting — the leakage guard
# --------------------------------------------------------------------------


def test_cross_fitting_covers_every_row(overconfident):
    y, p = overconfident
    out_of_fold = cross_fitted_probabilities(p, y, "isotonic", RecalibrationConfig())
    assert out_of_fold.shape == p.shape
    assert np.isfinite(out_of_fold).all()


def test_cross_fitted_probabilities_differ_from_in_sample_ones(overconfident):
    """The whole point: a row's own outcome must not have shaped its own score.

    Isotonic regression fits the calibration sample closely, so in-sample and
    out-of-fold values diverge visibly. Were they identical, cross-fitting
    would not be happening and the threshold would be tuned on a fit that has
    already seen the answers.
    """
    y, p = overconfident
    in_sample = build_calibrator("isotonic").fit(p, y).transform(p)
    out_of_fold = cross_fitted_probabilities(p, y, "isotonic", RecalibrationConfig())
    assert np.mean(np.abs(in_sample - out_of_fold)) > 1e-3


def test_identity_is_unaffected_by_cross_fitting(overconfident):
    """A map with nothing to fit cannot leak; a useful sanity anchor."""
    y, p = overconfident
    np.testing.assert_allclose(
        cross_fitted_probabilities(p, y, "identity", RecalibrationConfig()), p
    )


def test_cross_fitting_is_deterministic(overconfident):
    y, p = overconfident
    config = RecalibrationConfig(random_state=17)
    np.testing.assert_allclose(
        cross_fitted_probabilities(p, y, "platt", config),
        cross_fitted_probabilities(p, y, "platt", config),
    )


def test_cross_fitting_rejects_single_class_input():
    with pytest.raises(ValueError, match="both classes"):
        cross_fitted_probabilities(np.linspace(0.1, 0.9, 50), np.zeros(50, dtype=int), "platt")


# --------------------------------------------------------------------------
# The wrapper and the end-to-end fit
# --------------------------------------------------------------------------


def test_calibrated_model_satisfies_the_scoring_protocol(synthetic_frame, synthetic_target):
    config = Config()
    splits = make_splits(synthetic_frame, synthetic_target, config.split)
    model = build_model("logistic_regression", config.model).fit(splits.X_train, splits.y_train)

    fitted = fit_calibration(model, splits, "platt", config.recalibration)
    wrapped = fitted.calibrated_model

    assert isinstance(wrapped, ScoringModel)
    assert wrapped.name == "logistic_regression+platt"
    probabilities = wrapped.predict_proba_bad(splits.X_test)
    assert probabilities.shape == (len(splits.X_test),)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert set(np.unique(wrapped.decide(splits.X_test, 0.3))) <= {0, 1}


def test_wrapping_an_unfitted_component_is_rejected(synthetic_frame, synthetic_target):
    model = build_model("logistic_regression")
    with pytest.raises(NotFittedError, match="must be fitted"):
        CalibratedModel(model, IdentityCalibrator().fit(np.array([0.2]), np.array([0])))

    model.fit(synthetic_frame, synthetic_target)
    with pytest.raises(NotFittedError, match="must be fitted"):
        CalibratedModel(model, IdentityCalibrator())


def test_fit_calibration_does_not_refit_the_base_model(synthetic_frame, synthetic_target):
    """Recalibration learns a probability map, never a new model."""
    config = Config()
    splits = make_splits(synthetic_frame, synthetic_target, config.split)
    model = build_model("logistic_regression", config.model).fit(splits.X_train, splits.y_train)
    before = model.predict_proba_bad(splits.X_test).copy()

    fit_calibration(model, splits, "isotonic", config.recalibration)

    np.testing.assert_allclose(model.predict_proba_bad(splits.X_test), before)


def test_fit_calibration_requires_a_fitted_model(synthetic_frame, synthetic_target):
    splits = make_splits(synthetic_frame, synthetic_target)
    with pytest.raises(RuntimeError, match="must be fitted"):
        fit_calibration(build_model("logistic_regression"), splits, "platt")


def test_fit_all_calibrations_puts_the_control_first(synthetic_frame, synthetic_target):
    config = Config()
    splits = make_splits(synthetic_frame, synthetic_target, config.split)
    model = build_model("logistic_regression", config.model).fit(splits.X_train, splits.y_train)
    results = fit_all_calibrations(model, splits, config.recalibration)
    assert [r.calibrator_name for r in results] == list(config.recalibration.methods)
    assert results[0].calibrator_name == "identity"


def test_recalibration_never_reorders_test_applicants(synthetic_frame, synthetic_target):
    """End-to-end restatement of monotonicity, on the partition that is reported."""
    config = Config()
    splits = make_splits(synthetic_frame, synthetic_target, config.split)
    model = build_model("gradient_boosting", config.model).fit(splits.X_train, splits.y_train)
    raw = model.predict_proba_bad(splits.X_test)
    order = np.argsort(raw)

    for result in fit_all_calibrations(model, splits, config.recalibration):
        corrected = result.calibrated_model.predict_proba_bad(splits.X_test)
        assert np.all(np.diff(corrected[order]) >= -1e-12), result.calibrator_name


def test_n_output_levels_exposes_the_resolution_loss(synthetic_frame, synthetic_target):
    """Identity keeps every level; isotonic collapses the scale into few steps."""
    config = Config()
    splits = make_splits(synthetic_frame, synthetic_target, config.split)
    model = build_model("gradient_boosting", config.model).fit(splits.X_train, splits.y_train)
    by_method = {
        r.calibrator_name: r for r in fit_all_calibrations(model, splits, config.recalibration)
    }
    assert by_method["identity"].n_output_levels == len(splits.X_calibration)
    assert by_method["isotonic"].n_output_levels < by_method["identity"].n_output_levels
