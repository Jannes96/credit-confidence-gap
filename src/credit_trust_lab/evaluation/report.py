"""Assemble one model's evaluation into a single serialisable record.

The threshold is chosen on the **calibration** partition and applied unchanged
to the test partition. Choosing it on the test data would be a subtle form of
the leakage the three-way split exists to prevent: the threshold is a fitted
parameter, and fitting it on the data used to report performance flatters the
result exactly as fitting a coefficient there would.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from credit_trust_lab.config import Config
from credit_trust_lab.data.splits import DataSplits
from credit_trust_lab.evaluation.calibration import (
    CalibrationMetrics,
    TrustGapRow,
    calibration_metrics,
    trust_gap_table,
)
from credit_trust_lab.evaluation.costs import (
    ThresholdResult,
    confusion_at,
    expected_cost,
    optimise_threshold,
    threshold_sensitivity,
)
from credit_trust_lab.evaluation.discrimination import (
    DiscriminationMetrics,
    Interval,
    bootstrap_interval,
    discrimination_metrics,
)
from credit_trust_lab.evaluation.null_model import (
    NullComparison,
    calibration_null_comparison,
    threshold_gap_null_comparison,
)
from credit_trust_lab.models.base import CreditModel, ScoringModel


@dataclass(frozen=True)
class PartitionEvaluation:
    """Metrics for one model on one partition.

    Attributes:
        partition: ``"calibration"`` or ``"test"``.
        discrimination: Ranking quality.
        calibration: Calibration quality.
        trust_gap: Confidence band versus realised accuracy.
        y_true: Labels, retained so figures can be redrawn without refitting.
        y_prob: Predicted P(bad credit).

    """

    partition: str
    discrimination: DiscriminationMetrics
    calibration: CalibrationMetrics
    trust_gap: tuple[TrustGapRow, ...]
    y_true: np.ndarray
    y_prob: np.ndarray


@dataclass(frozen=True)
class ModelEvaluation:
    """Everything reported about one model.

    Attributes:
        model_name: Registry name of the model.
        calibration_partition: Evaluation on the calibration set.
        test_partition: Evaluation on the held-out test set.
        threshold: Threshold chosen on the calibration partition.
        test_cost_at_chosen_threshold: Realised test cost at that threshold.
        test_cost_per_case: The same, per decision.
        test_confusion: Test confusion counts at that threshold.
        roc_auc_interval: Bootstrap interval for test ROC-AUC.
        pr_auc_interval: Bootstrap interval for test PR-AUC.
        sensitivity: Threshold results across the cost-ratio sweep, on the
            calibration partition.
        test_calibration_null: Each test calibration metric held against what a
            perfectly calibrated model would have produced at this sample size.
            Without it an ECE of 0.06 reads as a defect when it is the floor.
        threshold_gap_null: The same treatment for the empirical-minus-
            theoretical threshold gap.

    """

    model_name: str
    test_calibration_null: dict[str, NullComparison]
    threshold_gap_null: NullComparison
    calibration_partition: PartitionEvaluation
    test_partition: PartitionEvaluation
    threshold: ThresholdResult
    test_cost_at_chosen_threshold: float
    test_cost_per_case: float
    test_confusion: dict[str, int]
    roc_auc_interval: Interval
    pr_auc_interval: Interval
    sensitivity: tuple[ThresholdResult, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary (raw score vectors excluded)."""
        return {
            "model": self.model_name,
            "threshold": self.threshold.as_dict(),
            "test": {
                "discrimination": self.test_partition.discrimination.as_dict(),
                "calibration": self.test_partition.calibration.as_dict(),
                "trust_gap": [row.__dict__ for row in self.test_partition.trust_gap],
                "roc_auc_ci": self.roc_auc_interval.as_dict(),
                "pr_auc_ci": self.pr_auc_interval.as_dict(),
                "cost_at_chosen_threshold": self.test_cost_at_chosen_threshold,
                "cost_per_case": self.test_cost_per_case,
                "confusion": self.test_confusion,
                "calibration_vs_null": {
                    name: comparison.as_dict()
                    for name, comparison in self.test_calibration_null.items()
                },
            },
            "threshold_gap_vs_null": self.threshold_gap_null.as_dict(),
            "calibration_partition": {
                "discrimination": self.calibration_partition.discrimination.as_dict(),
                "calibration": self.calibration_partition.calibration.as_dict(),
            },
            "cost_ratio_sensitivity": [result.as_dict() for result in self.sensitivity],
        }


def _evaluate_partition(
    partition: str, y_true: np.ndarray, y_prob: np.ndarray, config: Config
) -> PartitionEvaluation:
    """Compute discrimination, calibration and trust-gap for one partition."""
    return PartitionEvaluation(
        partition=partition,
        discrimination=discrimination_metrics(y_true, y_prob),
        calibration=calibration_metrics(y_true, y_prob, config.calibration),
        trust_gap=trust_gap_table(y_true, y_prob, config.calibration),
        y_true=np.asarray(y_true),
        y_prob=np.asarray(y_prob),
    )


def evaluate_model(
    model: ScoringModel,
    splits: DataSplits,
    config: Config | None = None,
    *,
    fit: bool = True,
    calibration_probabilities: np.ndarray | None = None,
) -> ModelEvaluation:
    """Fit a model on the training partition and evaluate it end to end.

    Args:
        model: Anything satisfying :class:`ScoringModel` — a raw model or a
            recalibrated one.
        splits: The three partitions.
        config: Full project configuration.
        fit: Set to ``False`` to evaluate an already-fitted model. Required
            for recalibrated models, which cannot be fitted from raw features.
        calibration_probabilities: Probabilities to use for the calibration
            partition instead of calling the model. Exists for recalibrated
            models: their calibrator was fitted on this partition, so calling
            the model here would return in-sample probabilities and the
            threshold would be tuned to a fit that does not generalise. Pass
            the out-of-fold probabilities from
            :func:`~credit_trust_lab.recalibration.fitting.fit_calibration`
            instead.

    Returns:
        The complete evaluation record.

    Raises:
        TypeError: If ``fit=True`` for something that is not a
            :class:`CreditModel`.
        ValueError: If ``calibration_probabilities`` has the wrong length.

    """
    config = config or Config()
    if fit:
        if not isinstance(model, CreditModel):
            raise TypeError(
                f"{type(model).__name__} cannot be fitted from raw features; "
                "pass fit=False and supply an already-fitted model."
            )
        model.fit(splits.X_train, splits.y_train)

    calib_true = np.asarray(splits.y_calibration)
    test_true = np.asarray(splits.y_test)

    if calibration_probabilities is None:
        calib_prob = model.predict_proba_bad(splits.X_calibration)
    else:
        calib_prob = np.asarray(calibration_probabilities, dtype=float).ravel()
        if calib_prob.shape != calib_true.shape:
            raise ValueError(
                f"calibration_probabilities has {calib_prob.shape} entries "
                f"but the calibration partition has {calib_true.shape}."
            )
    test_prob = model.predict_proba_bad(splits.X_test)

    # The threshold is a parameter fitted on the calibration partition.
    threshold = optimise_threshold(calib_true, calib_prob, config.cost)

    return ModelEvaluation(
        model_name=model.name,
        test_calibration_null=calibration_null_comparison(
            test_true, test_prob, config.calibration, config.null_model
        ),
        threshold_gap_null=threshold_gap_null_comparison(
            calib_true, calib_prob, config.cost, config.null_model
        ),
        calibration_partition=_evaluate_partition("calibration", calib_true, calib_prob, config),
        test_partition=_evaluate_partition("test", test_true, test_prob, config),
        threshold=threshold,
        test_cost_at_chosen_threshold=expected_cost(
            test_true, test_prob, threshold.threshold, config.cost
        ),
        test_cost_per_case=expected_cost(test_true, test_prob, threshold.threshold, config.cost)
        / test_true.size,
        test_confusion=confusion_at(test_true, test_prob, threshold.threshold).__dict__,
        roc_auc_interval=bootstrap_interval(test_true, test_prob, roc_auc_score, config.bootstrap),
        pr_auc_interval=bootstrap_interval(
            test_true, test_prob, average_precision_score, config.bootstrap
        ),
        sensitivity=tuple(threshold_sensitivity(calib_true, calib_prob, config.cost)),
    )


def write_json(payload: object, path: Path) -> None:
    """Write a payload as indented JSON, creating parent directories.

    Args:
        payload: Any JSON-serialisable object.
        path: Destination file.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
