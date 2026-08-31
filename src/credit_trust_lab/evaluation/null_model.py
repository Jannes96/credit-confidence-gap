"""What a *perfectly calibrated* model would score on a sample this small.

The motivating observation, found while interpreting the step 1 results: on a
200-row test partition split into 10 quantile bins, every bin holds 20
observations. At a true rate of 15 %, the standard error of an observed bin
frequency is about 8 percentage points, so the mean absolute gap between stated
and observed probability is roughly 6 points **even when the model is exactly
right**. An ECE of 0.06 measured on that sample is therefore not evidence of
miscalibration. It is the measurement floor.

This module makes that floor a first-class, reported quantity rather than
something a reader has to work out for themselves.

Method
------
Take the model's stated probabilities ``p`` and draw outcomes from them:
``y ~ Bernoulli(p)``. Those synthetic outcomes are, by construction, generated
by a perfectly calibrated process with exactly this score distribution and
exactly this sample size. Recomputing the metric across many such draws gives
its null distribution. Where the observed value falls in that distribution is
the honest answer to "is this model measurably miscalibrated?".

This is a parametric bootstrap under the null hypothesis of perfect
calibration. It says nothing about whether the probabilities are *useful* —
a constant predictor at the base rate is perfectly calibrated and worthless.
Discrimination answers that separately, which is why both are always reported.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from credit_trust_lab.config import CalibrationConfig, NullModelConfig
from credit_trust_lab.evaluation.calibration import (
    PooledTrustGapRow,
    TrustGapRow,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    trust_gap_table,
    weighted_band_gap,
)
from credit_trust_lab.evaluation.costs import CostConfig, optimise_threshold


@dataclass(frozen=True)
class NullComparison:
    """One metric, held against what perfect calibration would have produced.

    Attributes:
        metric: Name of the metric compared.
        observed: Value measured on the real outcomes.
        null_median: Median under perfect calibration at this sample size.
        null_lower: 5th percentile of the null distribution.
        null_upper: 95th percentile of the null distribution.
        percentile: Share of null draws below ``observed``. For an error
            metric, a small percentile means the model looks *better* than a
            perfectly calibrated one would typically look — which is possible
            purely by chance and is itself a sign the sample is too small.
        two_sided_p: Share of null draws at least as extreme as ``observed``,
            measured as distance from the null median. The usual reading
            applies: large means indistinguishable from perfect calibration.
        n_resamples: Number of null draws.
        n_samples: Rows the metric was computed on.

    """

    metric: str
    observed: float
    null_median: float
    null_lower: float
    null_upper: float
    percentile: float
    two_sided_p: float
    n_resamples: int
    n_samples: int

    @property
    def is_distinguishable(self) -> bool:
        """Whether the observed value lies outside the null 90 % interval.

        Deliberately not called ``is_significant``: this is a screening
        statement about measurement resolution, not a hypothesis test with a
        pre-registered level.
        """
        return not (self.null_lower <= self.observed <= self.null_upper)

    def as_dict(self) -> dict[str, object]:
        """Return a serialisable dictionary including the derived verdict."""
        return asdict(self) | {"is_distinguishable": self.is_distinguishable}

    def __str__(self) -> str:
        """One-line summary suitable for console output."""
        verdict = "outside" if self.is_distinguishable else "within"
        return (
            f"{self.metric}={self.observed:.3f} "
            f"(null median {self.null_median:.3f}, "
            f"90% [{self.null_lower:.3f}, {self.null_upper:.3f}], {verdict} noise)"
        )


def _null_draws(
    y_prob: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    config: NullModelConfig,
) -> np.ndarray:
    """Draw perfectly calibrated outcomes and recompute ``metric`` on each."""
    rng = np.random.default_rng(config.random_state)
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    draws = np.empty(config.n_resamples, dtype=float)
    for i in range(config.n_resamples):
        draws[i] = metric(rng.binomial(1, y_prob), y_prob)
    return draws


def _compare(
    metric_name: str,
    observed: float,
    draws: np.ndarray,
    n_samples: int,
    config: NullModelConfig,
) -> NullComparison:
    """Position an observed value inside its null distribution."""
    median = float(np.median(draws))
    distance = abs(observed - median)
    return NullComparison(
        metric=metric_name,
        observed=float(observed),
        null_median=median,
        null_lower=float(np.quantile(draws, 0.05)),
        null_upper=float(np.quantile(draws, 0.95)),
        percentile=float((draws < observed).mean()),
        two_sided_p=float((np.abs(draws - median) >= distance).mean()),
        n_resamples=config.n_resamples,
        n_samples=n_samples,
    )


def calibration_null_comparison(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    calibration_config: CalibrationConfig | None = None,
    config: NullModelConfig | None = None,
) -> dict[str, NullComparison]:
    """Hold ECE, MCE and the Brier score against perfect calibration.

    Args:
        y_true: Binary labels, 1 = bad credit.
        y_prob: Predicted P(bad credit).
        calibration_config: Binning used for ECE and MCE. The null must be
            computed with the *same* binning as the observed value, since the
            floor depends on how many observations land in each bin.
        config: Resample count and seed.

    Returns:
        One comparison per metric, keyed by metric name.

    """
    calibration_config = calibration_config or CalibrationConfig()
    config = config or NullModelConfig()
    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob, dtype=float).ravel()

    metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
        "expected_calibration_error": lambda t, p: expected_calibration_error(
            t, p, calibration_config
        ),
        "maximum_calibration_error": lambda t, p: maximum_calibration_error(
            t, p, calibration_config
        ),
        "brier_score": brier_score,
    }

    return {
        name: _compare(
            name, metric(y_true, y_prob), _null_draws(y_prob, metric, config), y_true.size, config
        )
        for name, metric in metrics.items()
    }


def threshold_gap_null_comparison(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_config: CostConfig | None = None,
    config: NullModelConfig | None = None,
) -> NullComparison:
    """Hold the empirical-minus-theoretical threshold gap against perfect calibration.

    Under perfect calibration the two thresholds should coincide *in
    expectation*. On a finite sample they will not, because the realised cost
    curve is a step function whose minimum wanders. This quantifies how far it
    wanders when nothing is wrong, so that an observed gap can be read against
    it instead of being taken at face value.

    Args:
        y_true: Binary labels, 1 = bad credit.
        y_prob: Predicted P(bad credit) on the partition the threshold is
            chosen from — the calibration partition, not test.
        cost_config: The cost matrix defining the theoretical threshold.
        config: Resample count and seed.

    Returns:
        The comparison for ``threshold_gap``.

    """
    cost_config = cost_config or CostConfig()
    config = config or NullModelConfig()
    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob, dtype=float).ravel()

    def gap(t: np.ndarray, p: np.ndarray) -> float:
        return optimise_threshold(t, p, cost_config).threshold_gap

    return _compare(
        "threshold_gap",
        gap(y_true, y_prob),
        _null_draws(y_prob, gap, config),
        y_true.size,
        config,
    )


def resolution_limit(
    y_prob: np.ndarray,
    calibration_config: CalibrationConfig | None = None,
    config: NullModelConfig | None = None,
) -> float:
    """Return the smallest ECE that would stand out from noise at this sample size.

    Reported in the console output and the README so that no calibration
    number in this project is quoted without the resolution of the instrument
    that produced it.

    Args:
        y_prob: Predicted probabilities, which fix the score distribution.
        calibration_config: Binning scheme.
        config: Resample count and seed.

    Returns:
        The 95th percentile of ECE under perfect calibration.

    """
    calibration_config = calibration_config or CalibrationConfig()
    config = config or NullModelConfig()
    draws = _null_draws(
        y_prob,
        lambda t, p: expected_calibration_error(t, p, calibration_config),
        config,
    )
    return float(np.quantile(draws, 0.95))


def pooled_trust_gap_null(
    observed: Sequence[PooledTrustGapRow],
    probability_tables: Sequence[np.ndarray],
    calibration_config: CalibrationConfig | None = None,
    config: NullModelConfig | None = None,
) -> dict[tuple[float, float], NullComparison]:
    """Hold each pooled confidence band against perfect calibration.

    Pooling raises the case count per band from roughly fifty to several
    hundred, which lowers the noise floor — but does not remove it. Without
    this comparison the pooled table would repeat, at a larger sample size,
    exactly the error that made the step 1 ECE figures unreadable: a number
    that looks like a measurement because it is precise.

    The null preserves the pooled structure. Every draw regenerates outcomes
    for *every* split from that split's own probabilities and pools the
    resulting tables the same way the observed value was pooled, so the
    overlap between splits is reproduced rather than assumed away.

    Args:
        observed: The pooled rows actually measured, from
            :func:`~credit_trust_lab.evaluation.calibration.pool_trust_gap`.
        probability_tables: Predicted P(bad credit) per split, in the same
            order the observed tables were pooled in.
        calibration_config: Supplies the confidence band edges. Must match the
            configuration the observed table used.
        config: Resample count and seed.

    Returns:
        One comparison per observed band, keyed by ``(lower, upper)``. Bands
        the null never produced are omitted.

    Raises:
        ValueError: If no probability tables are supplied.

    """
    calibration_config = calibration_config or CalibrationConfig()
    config = config or NullModelConfig()
    if not probability_tables:
        raise ValueError("Cannot build a null for an empty collection of splits.")

    rng = np.random.default_rng(config.random_state)
    per_band_draws: dict[tuple[float, float], list[float]] = {}

    for _ in range(config.n_resamples):
        grouped: dict[tuple[float, float], list[TrustGapRow]] = {}
        for probabilities in probability_tables:
            for row in trust_gap_table(
                rng.binomial(1, probabilities), probabilities, calibration_config
            ):
                grouped.setdefault((row.lower, row.upper), []).append(row)
        for band, rows in grouped.items():
            per_band_draws.setdefault(band, []).append(weighted_band_gap(rows)[2])

    comparisons: dict[tuple[float, float], NullComparison] = {}
    for pooled_row in observed:
        band = (pooled_row.lower, pooled_row.upper)
        draws = per_band_draws.get(band)
        if draws is None:
            continue
        comparisons[band] = _compare(
            f"pooled_trust_gap_{band[0]:.1f}_{band[1]:.1f}",
            pooled_row.gap,
            np.asarray(draws),
            pooled_row.count,
            config,
        )
    return comparisons
