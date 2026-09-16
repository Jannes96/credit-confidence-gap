"""Cost-based decision thresholds.

A threshold of 0.5 is the right answer to a question nobody in lending asks:
it minimises the error *count*, treating a missed default and a rejected good
customer as equally bad. This module derives the threshold from an explicit
cost matrix instead.

Two thresholds are computed and deliberately reported together:

* The **theoretical threshold** ``t* = C_FP / (C_FP + C_FN)``, which is optimal
  if — and only if — the predicted probabilities are calibrated. Under the
  default 5:1 Statlog costs, ``t* = 0.167``.
* The **empirical threshold**, found by sweeping the observed scores and
  minimising realised cost.

Their distance is a calibration diagnostic in decision units. If a model's
probabilities were trustworthy the two would coincide; when they diverge, the
gap says how much a decision-maker would lose by taking the model's stated
probabilities at face value. That is the guiding question of this project,
expressed in money rather than in a diagram.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from credit_confidence_gap.config import CostConfig


@dataclass(frozen=True)
class ConfusionCounts:
    """Confusion matrix in the project's orientation (positive = bad credit).

    Attributes:
        true_negative: Good credit correctly accepted.
        false_positive: Good credit wrongly refused.
        false_negative: Bad credit wrongly accepted — the expensive error.
        true_positive: Bad credit correctly refused.

    """

    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

    @property
    def n(self) -> int:
        """Total number of decisions."""
        return self.true_negative + self.false_positive + self.false_negative + self.true_positive


@dataclass(frozen=True)
class ThresholdResult:
    """Outcome of threshold optimisation on one partition.

    Attributes:
        threshold: Cost-minimising threshold found by sweeping the scores.
        theoretical_threshold: ``C_FP / (C_FP + C_FN)``, optimal under perfect
            calibration.
        threshold_gap: ``threshold - theoretical_threshold``.
        total_cost: Realised cost at the chosen threshold.
        cost_per_case: ``total_cost / n``.
        cost_at_half: Realised cost at the naive 0.5 threshold, for contrast.
        cost_at_theoretical: Realised cost at the theoretical threshold.
        confusion: Confusion counts at the chosen threshold.
        cost_ratio: ``C_FN / C_FP``, the assumption the result hangs on.

    """

    threshold: float
    theoretical_threshold: float
    threshold_gap: float
    total_cost: float
    cost_per_case: float
    cost_at_half: float
    cost_at_theoretical: float
    confusion: ConfusionCounts
    cost_ratio: float

    def as_dict(self) -> dict[str, object]:
        """Return a serialisable dictionary."""
        payload = asdict(self)
        payload["confusion"] = asdict(self.confusion)
        return payload


def theoretical_threshold(config: CostConfig) -> float:
    """Bayes-optimal threshold under perfectly calibrated probabilities.

    Refusing is preferable when ``p * C_TP + (1-p) * C_FP <= p * C_FN + (1-p) * C_TN``,
    which rearranges to ``p >= (C_FP - C_TN) / ((C_FN - C_TP) + (C_FP - C_TN))``.

    Args:
        config: The cost matrix.

    Returns:
        The threshold in (0, 1).

    """
    gain_from_refusing_bad = config.cost_false_negative - config.cost_true_positive
    loss_from_refusing_good = config.cost_false_positive - config.cost_true_negative
    return float(loss_from_refusing_good / (gain_from_refusing_bad + loss_from_refusing_good))


def confusion_at(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> ConfusionCounts:
    """Count decision outcomes at a given threshold.

    Args:
        y_true: Binary labels, 1 = bad credit.
        y_prob: Predicted P(bad credit).
        threshold: Score at or above which the credit is refused.

    Returns:
        The four confusion counts.

    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    refused = y_prob >= threshold
    return ConfusionCounts(
        true_negative=int(np.sum(~refused & (y_true == 0))),
        false_positive=int(np.sum(refused & (y_true == 0))),
        false_negative=int(np.sum(~refused & (y_true == 1))),
        true_positive=int(np.sum(refused & (y_true == 1))),
    )


def expected_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    config: CostConfig | None = None,
) -> float:
    """Total realised cost of deciding at ``threshold``.

    Args:
        y_true: Binary labels, 1 = bad credit.
        y_prob: Predicted P(bad credit).
        threshold: Score at or above which the credit is refused.
        config: The cost matrix.

    Returns:
        Total cost in the matrix's units.

    """
    config = config or CostConfig()
    counts = confusion_at(y_true, y_prob, threshold)
    return float(
        counts.true_negative * config.cost_true_negative
        + counts.false_positive * config.cost_false_positive
        + counts.false_negative * config.cost_false_negative
        + counts.true_positive * config.cost_true_positive
    )


def cost_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    config: CostConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Realised cost across all candidate thresholds.

    Candidates are the observed scores plus the endpoints, since cost is a step
    function of the threshold that can only change where a score lies.

    Args:
        y_true: Binary labels, 1 = bad credit.
        y_prob: Predicted P(bad credit).
        config: The cost matrix.

    Returns:
        ``(thresholds, costs)``, thresholds ascending.

    """
    config = config or CostConfig()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    candidates = np.unique(np.concatenate(([0.0], y_prob, [1.0 + 1e-9])))
    costs = np.array([expected_cost(y_true, y_prob, t, config) for t in candidates])
    return candidates, costs


def optimise_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    config: CostConfig | None = None,
) -> ThresholdResult:
    """Find the cost-minimising threshold by exhaustive sweep.

    Ties are broken towards the *higher* threshold, i.e. towards accepting.
    Some other choice would be equally defensible; the point is that the rule
    is stated rather than left to floating-point ordering.

    Args:
        y_true: Binary labels, 1 = bad credit.
        y_prob: Predicted P(bad credit).
        config: The cost matrix.

    Returns:
        The chosen threshold with its costs, its theoretical counterpart, and
        the confusion counts it implies.

    """
    config = config or CostConfig()
    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if y_true.size == 0:
        raise ValueError("Cannot optimise a threshold on an empty sample.")

    candidates, costs = cost_curve(y_true, y_prob, config)
    minimum = costs.min()
    best = float(candidates[np.flatnonzero(costs == minimum)[-1]])
    best = min(best, 1.0)

    theoretical = theoretical_threshold(config)
    return ThresholdResult(
        threshold=best,
        theoretical_threshold=theoretical,
        threshold_gap=best - theoretical,
        total_cost=float(minimum),
        cost_per_case=float(minimum / y_true.size),
        cost_at_half=expected_cost(y_true, y_prob, 0.5, config),
        cost_at_theoretical=expected_cost(y_true, y_prob, theoretical, config),
        confusion=confusion_at(y_true, y_prob, best),
        cost_ratio=config.cost_false_negative / config.cost_false_positive,
    )


def threshold_sensitivity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    config: CostConfig | None = None,
) -> list[ThresholdResult]:
    """Re-optimise the threshold across a range of FN/FP cost ratios.

    The 5:1 ratio is a convention inherited from the Statlog problem, not a
    measurement. Sweeping it shows how much of the recommended decision is
    driven by the data and how much by that inherited assumption.

    Args:
        y_true: Binary labels, 1 = bad credit.
        y_prob: Predicted P(bad credit).
        config: Supplies ``sensitivity_ratios`` and the FP cost held fixed.

    Returns:
        One result per ratio, in the order given by ``sensitivity_ratios``.

    """
    config = config or CostConfig()
    results: list[ThresholdResult] = []
    for ratio in config.sensitivity_ratios:
        variant = CostConfig(
            cost_false_negative=config.cost_false_positive * ratio,
            cost_false_positive=config.cost_false_positive,
            cost_true_negative=config.cost_true_negative,
            cost_true_positive=config.cost_true_positive,
            sensitivity_ratios=config.sensitivity_ratios,
        )
        results.append(optimise_threshold(y_true, y_prob, variant))
    return results
