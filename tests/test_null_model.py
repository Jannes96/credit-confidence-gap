"""Tests for the perfect-calibration null distribution.

This is the instrument that decides whether any calibration number in the
project means anything, so it is pinned against cases whose answer is known by
construction: data generated to be perfectly calibrated must land inside the
null band, and data generated to be badly wrong must land outside it.
"""

from __future__ import annotations

import numpy as np
import pytest

from credit_confidence_gap.config import CalibrationConfig, NullModelConfig
from credit_confidence_gap.evaluation.null_model import (
    calibration_null_comparison,
    resolution_limit,
    threshold_gap_null_comparison,
)

# Small resample counts keep the suite fast; the assertions are all coarse
# enough that 300 draws settle them.
FAST = NullModelConfig(n_resamples=300, random_state=4242)


@pytest.fixture
def calibrated_200(rng):
    """Return a 200-row perfectly calibrated sample, the size the project uses."""
    p = rng.uniform(0.02, 0.9, size=200)
    return rng.binomial(1, p), p


def test_perfectly_calibrated_data_is_rarely_flagged():
    """The instrument's false-positive rate must match its stated 90% band.

    Asserting that a single perfectly calibrated sample falls inside the band
    would be a test that fails 10% of the time by construction — the band is a
    90% interval. The testable claim is about the *rate* across many samples,
    which is also the property that matters: an instrument flagging every
    honest model would be useless.
    """
    rng = np.random.default_rng(2024)
    flagged = 0
    trials = 20
    for _ in range(trials):
        p = rng.uniform(0.02, 0.9, size=200)
        y = rng.binomial(1, p)
        comparison = calibration_null_comparison(y, p, CalibrationConfig(), FAST)[
            "expected_calibration_error"
        ]
        flagged += int(comparison.is_distinguishable)
    assert flagged / trials < 0.3, f"flagged {flagged}/{trials} honest models"


def test_badly_miscalibrated_data_is_distinguishable(rng):
    """Stated probabilities near zero while half the sample defaults."""
    y_prob = np.full(200, 0.02)
    y_true = rng.binomial(1, 0.5, size=200)
    comparison = calibration_null_comparison(y_true, y_prob, CalibrationConfig(), FAST)[
        "expected_calibration_error"
    ]
    assert comparison.is_distinguishable
    assert comparison.observed > comparison.null_upper


def test_the_noise_floor_at_n200_is_substantial(calibrated_200):
    """The finding that motivated this module, pinned as a test.

    Ten quantile bins over 200 rows leave 20 observations each, so a perfectly
    calibrated model still scores an ECE around 0.06. Any project quoting an
    ECE of that size on this sample as evidence of miscalibration is reading
    noise.
    """
    _, y_prob = calibrated_200
    floor = resolution_limit(y_prob, CalibrationConfig(), FAST)
    assert 0.03 < floor < 0.15


def test_more_data_lowers_the_noise_floor(rng):
    """The floor must fall as the sample grows, or it is not measuring resolution."""
    small = rng.uniform(0.05, 0.95, size=200)
    large = rng.uniform(0.05, 0.95, size=5000)
    assert resolution_limit(large, CalibrationConfig(), FAST) < resolution_limit(
        small, CalibrationConfig(), FAST
    )


def test_all_three_metrics_are_compared():
    y_prob = np.linspace(0.05, 0.95, 200)
    y_true = (np.arange(200) % 3 == 0).astype(int)
    comparisons = calibration_null_comparison(y_true, y_prob, CalibrationConfig(), FAST)
    assert set(comparisons) == {
        "expected_calibration_error",
        "maximum_calibration_error",
        "brier_score",
    }
    for comparison in comparisons.values():
        assert comparison.null_lower <= comparison.null_median <= comparison.null_upper
        assert 0.0 <= comparison.percentile <= 1.0
        assert comparison.n_samples == 200


def test_null_is_computed_with_the_same_binning_as_the_observed_value(calibrated_200):
    """A null drawn under different binning would not be a comparison at all."""
    y_true, y_prob = calibrated_200
    coarse = calibration_null_comparison(
        y_true, y_prob, CalibrationConfig(n_bins=4, binning="uniform"), FAST
    )["expected_calibration_error"]
    fine = calibration_null_comparison(
        y_true, y_prob, CalibrationConfig(n_bins=20, binning="quantile"), FAST
    )["expected_calibration_error"]
    # More bins mean fewer observations each, so a higher floor.
    assert fine.null_median > coarse.null_median


def test_is_distinguishable_matches_the_interval():
    y_prob = np.full(400, 0.10)
    y_true = np.zeros(400, dtype=int)
    y_true[:200] = 1  # 50% observed against 10% stated
    comparison = calibration_null_comparison(y_true, y_prob, CalibrationConfig(), FAST)[
        "expected_calibration_error"
    ]
    assert comparison.is_distinguishable
    assert not (comparison.null_lower <= comparison.observed <= comparison.null_upper)


def test_threshold_gap_null_is_centred_near_zero(calibrated_200):
    """Under perfect calibration the two thresholds should agree in expectation."""
    y_true, y_prob = calibrated_200
    comparison = threshold_gap_null_comparison(y_true, y_prob, config=FAST)
    assert abs(comparison.null_median) < 0.15
    assert comparison.null_lower < comparison.null_upper


def test_results_are_deterministic_under_a_fixed_seed(calibrated_200):
    y_true, y_prob = calibrated_200
    first = calibration_null_comparison(y_true, y_prob, CalibrationConfig(), FAST)
    second = calibration_null_comparison(y_true, y_prob, CalibrationConfig(), FAST)
    assert first["brier_score"].null_median == second["brier_score"].null_median


def test_comparison_serialises_with_its_verdict(calibrated_200):
    y_true, y_prob = calibrated_200
    payload = calibration_null_comparison(y_true, y_prob, CalibrationConfig(), FAST)[
        "brier_score"
    ].as_dict()
    assert "is_distinguishable" in payload
    assert payload["metric"] == "brier_score"
