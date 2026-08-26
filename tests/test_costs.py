"""Tests for the cost matrix and threshold logic.

The key property: under the default 5:1 costs the threshold must land far below
0.5, and on well calibrated scores the empirical optimum must agree with the
analytical one. If those two ever drift apart in the real results, that is a
finding about the model, not a bug — which is exactly why the arithmetic itself
is pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from credit_trust_lab.config import CostConfig
from credit_trust_lab.evaluation.costs import (
    confusion_at,
    cost_curve,
    expected_cost,
    optimise_threshold,
    theoretical_threshold,
    threshold_sensitivity,
)


def test_theoretical_threshold_for_the_statlog_default():
    """C_FP / (C_FP + C_FN) = 1 / 6."""
    assert theoretical_threshold(CostConfig()) == pytest.approx(1 / 6)


def test_symmetric_costs_reproduce_the_naive_half():
    config = CostConfig(cost_false_negative=1.0, cost_false_positive=1.0)
    assert theoretical_threshold(config) == pytest.approx(0.5)


def test_asymmetric_costs_push_the_threshold_below_half():
    for ratio in (2.0, 5.0, 10.0, 20.0):
        config = CostConfig(cost_false_negative=ratio, cost_false_positive=1.0)
        assert theoretical_threshold(config) < 0.5


def test_threshold_is_monotone_in_the_cost_ratio():
    """The more a missed default costs, the readier the model must be to refuse."""
    thresholds = [
        theoretical_threshold(CostConfig(cost_false_negative=r, cost_false_positive=1.0))
        for r in (1.0, 2.0, 5.0, 10.0, 20.0)
    ]
    assert thresholds == sorted(thresholds, reverse=True)


def test_correct_rewards_shift_the_threshold():
    """Non-zero payoffs for correct decisions must enter the formula."""
    config = CostConfig(
        cost_false_negative=5.0,
        cost_false_positive=1.0,
        cost_true_negative=-1.0,  # a good loan earns margin
        cost_true_positive=0.0,
    )
    assert theoretical_threshold(config) == pytest.approx(2.0 / 7.0)


def test_cost_config_rejects_an_incoherent_matrix():
    with pytest.raises(ValueError, match="cost_false_negative"):
        CostConfig(cost_false_negative=1.0, cost_true_positive=2.0)
    with pytest.raises(ValueError, match="cost_false_positive"):
        CostConfig(cost_false_positive=1.0, cost_true_negative=2.0)


def test_confusion_counts_use_the_project_orientation():
    """Positive = bad credit; refuse when p >= threshold."""
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.1, 0.9, 0.1])
    counts = confusion_at(y_true, y_prob, 0.5)
    assert counts.true_positive == 1  # bad credit, refused
    assert counts.false_negative == 1  # bad credit, accepted
    assert counts.false_positive == 1  # good credit, refused
    assert counts.true_negative == 1
    assert counts.n == 4


def test_threshold_is_inclusive_at_the_boundary():
    y_true = np.array([1])
    y_prob = np.array([0.3])
    assert confusion_at(y_true, y_prob, 0.3).true_positive == 1
    assert confusion_at(y_true, y_prob, 0.30001).false_negative == 1


def test_expected_cost_is_a_weighted_confusion_count():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.1, 0.9, 0.1])
    config = CostConfig(cost_false_negative=5.0, cost_false_positive=1.0)
    # one FN at 5 plus one FP at 1
    assert expected_cost(y_true, y_prob, 0.5, config) == pytest.approx(6.0)


def test_threshold_zero_refuses_everyone(calibrated_predictions):
    y_true, y_prob = calibrated_predictions
    counts = confusion_at(y_true, y_prob, 0.0)
    assert counts.false_negative == 0
    assert counts.true_positive == int(y_true.sum())


def test_threshold_above_one_accepts_everyone(calibrated_predictions):
    y_true, y_prob = calibrated_predictions
    counts = confusion_at(y_true, y_prob, 1.01)
    assert counts.true_positive == 0
    assert counts.false_negative == int(y_true.sum())


def test_optimal_threshold_matches_theory_on_calibrated_scores(calibrated_predictions):
    """The decisive test: with honest probabilities the two thresholds agree.

    Any divergence in the real results is therefore attributable to the model's
    calibration and not to the optimiser.
    """
    y_true, y_prob = calibrated_predictions
    result = optimise_threshold(y_true, y_prob, CostConfig())
    assert result.threshold == pytest.approx(result.theoretical_threshold, abs=0.02)
    assert abs(result.threshold_gap) < 0.02


def test_optimum_is_at_least_as_good_as_any_other_threshold(calibrated_predictions):
    y_true, y_prob = calibrated_predictions
    result = optimise_threshold(y_true, y_prob)
    for candidate in np.linspace(0.0, 1.0, 51):
        assert result.total_cost <= expected_cost(y_true, y_prob, candidate) + 1e-9


def test_naive_half_is_more_expensive_than_the_cost_optimal_threshold(calibrated_predictions):
    """The whole reason 0.5 is rejected, asserted rather than argued."""
    y_true, y_prob = calibrated_predictions
    result = optimise_threshold(y_true, y_prob, CostConfig())
    assert result.total_cost < result.cost_at_half


def test_cost_curve_covers_every_observed_score():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.2, 0.4, 0.6, 0.8])
    grid, costs = cost_curve(y_true, y_prob)
    assert grid.shape == costs.shape
    assert np.all(np.diff(grid) > 0)
    assert set(y_prob).issubset(set(grid))


def test_reported_cost_matches_a_recomputation(calibrated_predictions):
    y_true, y_prob = calibrated_predictions
    result = optimise_threshold(y_true, y_prob)
    assert result.total_cost == pytest.approx(expected_cost(y_true, y_prob, result.threshold))
    assert result.cost_per_case == pytest.approx(result.total_cost / y_true.size)


def test_sensitivity_sweep_returns_one_result_per_ratio():
    config = CostConfig()
    y_true = np.array([1, 0, 1, 0, 1, 0, 0, 0])
    y_prob = np.array([0.8, 0.2, 0.7, 0.3, 0.6, 0.1, 0.4, 0.05])
    results = threshold_sensitivity(y_true, y_prob, config)
    assert len(results) == len(config.sensitivity_ratios)
    assert [r.cost_ratio for r in results] == list(config.sensitivity_ratios)


def test_sensitivity_thresholds_do_not_increase_with_the_ratio(calibrated_predictions):
    y_true, y_prob = calibrated_predictions
    thresholds = [r.threshold for r in threshold_sensitivity(y_true, y_prob)]
    assert thresholds == sorted(thresholds, reverse=True)


def test_empty_sample_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        optimise_threshold(np.array([], dtype=int), np.array([]))
