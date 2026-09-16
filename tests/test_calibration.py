"""Tests for the calibration metrics.

Calibration code is easy to write and hard to check by eye, so it is pinned
here against cases whose correct answer is known analytically: a perfectly
calibrated sample, a maximally overconfident one, and a constant predictor.
"""

from __future__ import annotations

import numpy as np
import pytest

from credit_confidence_gap.config import CalibrationConfig
from credit_confidence_gap.evaluation.calibration import (
    brier_score,
    calibration_metrics,
    expected_calibration_error,
    reliability_bins,
    trust_gap_table,
)


def test_brier_score_of_a_perfect_predictor_is_zero():
    y = np.array([0, 1, 1, 0])
    assert brier_score(y, y.astype(float)) == 0.0


def test_brier_score_of_a_constant_half_predictor():
    y = np.array([0, 1, 1, 0])
    assert brier_score(y, np.full(4, 0.5)) == pytest.approx(0.25)


def test_brier_score_of_a_maximally_wrong_predictor_is_one():
    y = np.array([0, 1, 1, 0])
    assert brier_score(y, 1.0 - y.astype(float)) == pytest.approx(1.0)


def test_ece_is_near_zero_for_a_calibrated_sample(calibrated_predictions):
    """Outcomes drawn from the stated probabilities must show almost no gap."""
    y_true, y_prob = calibrated_predictions
    assert expected_calibration_error(y_true, y_prob) < 0.02


def test_ece_is_one_for_a_maximally_overconfident_predictor():
    y_true = np.array([0] * 50 + [1] * 50)
    y_prob = np.where(y_true == 1, 0.0, 1.0)  # confidently, uniformly wrong
    assert expected_calibration_error(y_true, y_prob) == pytest.approx(1.0)


def test_ece_matches_a_hand_computed_example():
    """Two occupied bins with known gaps, weighted by their counts.

    Ten predictions at 0.1 of which 3 are positive: gap |0.1 - 0.3| = 0.2.
    Ten predictions at 0.9 of which 9 are positive: gap |0.9 - 0.9| = 0.0.
    Count-weighted ECE = (10*0.2 + 10*0.0) / 20 = 0.1.
    """
    y_prob = np.concatenate([np.full(10, 0.1), np.full(10, 0.9)])
    y_true = np.array([1, 1, 1] + [0] * 7 + [1] * 9 + [0])
    config = CalibrationConfig(n_bins=2, binning="uniform")
    assert expected_calibration_error(y_true, y_prob, config) == pytest.approx(0.1)


def test_signed_error_distinguishes_over_from_underconfidence():
    y_true = np.zeros(100, dtype=int)
    y_true[:10] = 1
    overstating = np.full(100, 0.4)  # says 40% risk, 10% observed
    understating = np.full(100, 0.01)

    over = calibration_metrics(y_true, overstating)
    under = calibration_metrics(y_true, understating)

    assert over.signed_calibration_error > 0
    assert under.signed_calibration_error < 0
    # The absolute ECE cannot tell these two failure directions apart.
    assert over.expected_calibration_error > 0
    assert under.expected_calibration_error > 0


def test_bins_partition_the_sample(calibrated_predictions):
    y_true, y_prob = calibrated_predictions
    bins = reliability_bins(y_true, y_prob)
    assert sum(b.count for b in bins) == y_true.size


def test_empty_bins_are_dropped_not_reported_as_zero():
    """A uniform binning over clustered scores must not emit unoccupied bins."""
    y_prob = np.full(50, 0.05)
    y_true = np.zeros(50, dtype=int)
    bins = reliability_bins(y_true, y_prob, CalibrationConfig(n_bins=10, binning="uniform"))
    assert len(bins) == 1
    assert all(b.count > 0 for b in bins)


def test_prediction_of_exactly_zero_is_not_dropped():
    y_prob = np.array([0.0, 0.0, 0.5, 1.0])
    y_true = np.array([0, 0, 1, 1])
    bins = reliability_bins(y_true, y_prob, CalibrationConfig(n_bins=4, binning="uniform"))
    assert sum(b.count for b in bins) == 4


def test_mce_is_at_least_ece(calibrated_predictions):
    y_true, y_prob = calibrated_predictions
    metrics = calibration_metrics(y_true, y_prob)
    assert metrics.maximum_calibration_error >= metrics.expected_calibration_error


def test_quantile_binning_spreads_clustered_scores(rng):
    """Quantile bins must stay populated where uniform bins would collapse."""
    y_prob = rng.uniform(0.0, 0.2, size=1000)
    y_true = rng.binomial(1, y_prob)
    quantile = reliability_bins(y_true, y_prob, CalibrationConfig(n_bins=10, binning="quantile"))
    uniform = reliability_bins(y_true, y_prob, CalibrationConfig(n_bins=10, binning="uniform"))
    assert len(quantile) > len(uniform)


def test_unknown_binning_is_rejected():
    with pytest.raises(ValueError, match="Unknown binning"):
        reliability_bins(np.array([0, 1]), np.array([0.1, 0.9]), CalibrationConfig(binning="magic"))


def test_trust_gap_bands_are_built_on_decision_confidence():
    """Confidence is max(p, 1-p), so no band below 0.5 can ever be occupied."""
    y_prob = np.array([0.02, 0.05, 0.5, 0.95, 0.98])
    y_true = np.array([0, 0, 1, 1, 1])
    rows = trust_gap_table(y_true, y_prob)
    assert all(row.lower >= 0.5 for row in rows)
    assert sum(row.count for row in rows) == y_true.size


def test_trust_gap_is_positive_when_confidence_outruns_accuracy():
    """95% stated confidence, 50% realised accuracy: a gap of about 0.45."""
    y_prob = np.array([0.95] * 10)
    y_true = np.array([1] * 5 + [0] * 5)
    rows = trust_gap_table(y_true, y_prob)
    assert len(rows) == 1
    assert rows[0].mean_confidence == pytest.approx(0.95)
    assert rows[0].accuracy == pytest.approx(0.5)
    assert rows[0].gap == pytest.approx(0.45)


def test_trust_gap_is_near_zero_for_a_calibrated_model(calibrated_predictions):
    y_true, y_prob = calibrated_predictions
    for row in trust_gap_table(y_true, y_prob):
        assert abs(row.gap) < 0.05, row


def test_shares_sum_to_one(calibrated_predictions):
    y_true, y_prob = calibrated_predictions
    rows = trust_gap_table(y_true, y_prob)
    assert sum(row.share for row in rows) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("y_true", "y_prob", "message"),
    [
        (np.array([0, 1]), np.array([0.5]), "Shape mismatch"),
        (np.array([], dtype=int), np.array([]), "empty"),
        (np.array([0, 2]), np.array([0.1, 0.2]), "only 0 and 1"),
        (np.array([0, 1]), np.array([0.1, 1.5]), r"\[0, 1\]"),
        (np.array([0, 1]), np.array([0.1, np.nan]), "NaN"),
    ],
)
def test_malformed_inputs_are_rejected(y_true, y_prob, message):
    with pytest.raises(ValueError, match=message):
        calibration_metrics(y_true, y_prob)
