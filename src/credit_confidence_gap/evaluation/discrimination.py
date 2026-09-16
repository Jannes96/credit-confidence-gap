"""Discrimination: how well the score orders applicants.

Reported first and weighted last. Discrimination is invariant to any monotone
transformation of the score, which means a model can achieve an excellent AUC
while its stated probabilities are systematically wrong. That gap is the
subject of :mod:`credit_confidence_gap.evaluation.calibration`; this module exists
so the two can be shown side by side.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from credit_confidence_gap.config import BootstrapConfig


@dataclass(frozen=True)
class Interval:
    """A percentile bootstrap interval.

    Attributes:
        point: Metric on the full sample.
        lower: Lower percentile bound.
        upper: Upper percentile bound.
        confidence_level: Nominal coverage, e.g. 0.95.

    """

    point: float
    lower: float
    upper: float
    confidence_level: float

    def as_dict(self) -> dict[str, float]:
        """Return the interval as a plain dictionary for serialisation."""
        return asdict(self)

    def __str__(self) -> str:
        """Format as ``point [lower, upper]``."""
        return f"{self.point:.3f} [{self.lower:.3f}, {self.upper:.3f}]"


@dataclass(frozen=True)
class DiscriminationMetrics:
    """Ranking quality of a score.

    Attributes:
        roc_auc: Area under the ROC curve.
        pr_auc: Average precision for the bad-credit class.
        prevalence: Share of bad credits, i.e. the PR-AUC of a random score.
            Reported alongside because PR-AUC is not comparable across samples
            with different base rates, and this sample's base rate is an
            artefact of the sampling design.
        n_samples: Rows the metrics were computed on.
        n_positive: Bad credits among them.

    """

    roc_auc: float
    pr_auc: float
    prevalence: float
    n_samples: int
    n_positive: int

    def as_dict(self) -> dict[str, float | int]:
        """Return the metrics as a plain dictionary for serialisation."""
        return asdict(self)


def _validate(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coerce and check a (labels, probabilities) pair."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}.")
    if y_true.size == 0:
        raise ValueError("Cannot evaluate an empty sample.")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("y_true must contain only 0 and 1.")
    if np.isnan(y_prob).any():
        raise ValueError("y_prob contains NaN.")
    if (y_prob < 0).any() or (y_prob > 1).any():
        raise ValueError("y_prob must lie in [0, 1].")
    return y_true, y_prob


def discrimination_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> DiscriminationMetrics:
    """Compute ROC-AUC and PR-AUC for the bad-credit class.

    Args:
        y_true: Binary labels, 1 = bad credit.
        y_prob: Predicted P(bad credit).

    Returns:
        The discrimination metrics.

    Raises:
        ValueError: If the inputs are malformed or contain a single class.

    """
    y_true, y_prob = _validate(y_true, y_prob)
    if len(np.unique(y_true)) < 2:
        raise ValueError("Both classes must be present to compute AUC.")
    return DiscriminationMetrics(
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        prevalence=float(y_true.mean()),
        n_samples=int(y_true.size),
        n_positive=int(y_true.sum()),
    )


def roc_curve_data(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, np.ndarray]:
    """Return ROC curve coordinates for plotting.

    Args:
        y_true: Binary labels, 1 = bad credit.
        y_prob: Predicted P(bad credit).

    Returns:
        Mapping with ``fpr``, ``tpr`` and ``thresholds``.

    """
    y_true, y_prob = _validate(y_true, y_prob)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}


def pr_curve_data(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, np.ndarray]:
    """Return precision-recall curve coordinates for plotting.

    Args:
        y_true: Binary labels, 1 = bad credit.
        y_prob: Predicted P(bad credit).

    Returns:
        Mapping with ``precision``, ``recall`` and ``thresholds``.

    """
    y_true, y_prob = _validate(y_true, y_prob)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return {"precision": precision, "recall": recall, "thresholds": thresholds}


def bootstrap_interval(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    config: BootstrapConfig | None = None,
) -> Interval:
    """Percentile bootstrap interval for any metric of ``(y_true, y_prob)``.

    Resampling is stratified by class, so every resample keeps the sample's
    300/700 design and no resample degenerates to a single class.

    With ~60 bad credits in a 200-row test partition, these intervals are wide.
    That is the finding, not a shortcoming of the procedure: a project about
    unwarranted confidence should not report its own metrics as point estimates.

    Args:
        y_true: Binary labels, 1 = bad credit.
        y_prob: Predicted P(bad credit).
        metric: Callable returning a scalar, e.g. ``roc_auc_score``.
        config: Resample count, coverage and seed.

    Returns:
        The point estimate with its interval.

    """
    config = config or BootstrapConfig()
    y_true, y_prob = _validate(y_true, y_prob)

    rng = np.random.default_rng(config.random_state)
    positive_idx = np.flatnonzero(y_true == 1)
    negative_idx = np.flatnonzero(y_true == 0)

    estimates = np.empty(config.n_resamples, dtype=float)
    for i in range(config.n_resamples):
        picked = np.concatenate(
            (
                rng.choice(positive_idx, size=positive_idx.size, replace=True),
                rng.choice(negative_idx, size=negative_idx.size, replace=True),
            )
        )
        estimates[i] = metric(y_true[picked], y_prob[picked])

    alpha = (1.0 - config.confidence_level) / 2.0
    return Interval(
        point=float(metric(y_true, y_prob)),
        lower=float(np.quantile(estimates, alpha)),
        upper=float(np.quantile(estimates, 1.0 - alpha)),
        confidence_level=config.confidence_level,
    )
