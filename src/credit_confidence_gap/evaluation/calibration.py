"""Calibration: whether the model's confidence deserves to be believed.

This is the module the project is built around. A score with excellent
discrimination can still assert "8% risk" for a group that defaults 25% of the
time; a human reading that number, especially alongside a plausible
explanation, has no way to notice. Measuring the gap is a precondition for
studying it.

Recalibration itself (Platt scaling, isotonic regression) is deliberately *not*
here yet — it belongs to the next step and will consume the calibration
partition that :mod:`credit_confidence_gap.data.splits` already reserves. What is
here is the measurement apparatus those methods will be judged by.

A standing caveat applies to every number this module produces. The sample is
stratified by design (300 bad, 700 good), so the base rate is a property of the
sampling frame and not of any lending population. A model calibrated against
this sample is calibrated to the *sample distribution*. Absolute default
probabilities must not be claimed from it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from credit_confidence_gap.config import BootstrapConfig, CalibrationConfig


@dataclass(frozen=True)
class ReliabilityBin:
    """One bin of a reliability diagram.

    Attributes:
        lower: Lower edge of the probability bin.
        upper: Upper edge of the probability bin.
        count: Rows falling into the bin.
        mean_predicted: Mean predicted probability within the bin.
        observed_frequency: Observed share of bad credits within the bin.

    """

    lower: float
    upper: float
    count: int
    mean_predicted: float
    observed_frequency: float

    @property
    def gap(self) -> float:
        """Signed confidence-minus-correctness gap; positive means overconfident."""
        return self.mean_predicted - self.observed_frequency


@dataclass(frozen=True)
class CalibrationMetrics:
    """Calibration summary for one model on one partition.

    Attributes:
        brier_score: Mean squared error of the probabilities. Proper, but mixes
            calibration and discrimination, which is why it is not reported alone.
        expected_calibration_error: Count-weighted mean absolute gap.
        maximum_calibration_error: Largest absolute gap over occupied bins.
        signed_calibration_error: Count-weighted *signed* gap. Its sign is the
            informative part: positive means the model overstates risk on
            average, negative means it understates it. The absolute ECE cannot
            distinguish these, yet they fail in opposite directions.
        mean_predicted: Mean predicted probability over the partition.
        observed_frequency: Observed share of bad credits in the partition.
        n_bins: Requested bin count.
        binning: ``"quantile"`` or ``"uniform"``.
        bins: The occupied bins backing a reliability diagram.

    """

    brier_score: float
    expected_calibration_error: float
    maximum_calibration_error: float
    signed_calibration_error: float
    mean_predicted: float
    observed_frequency: float
    n_bins: int
    binning: str
    bins: tuple[ReliabilityBin, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a serialisable dictionary, bins included."""
        payload = asdict(self)
        payload["bins"] = [asdict(b) | {"gap": b.gap} for b in self.bins]
        return payload


@dataclass(frozen=True)
class TrustGapRow:
    """Confidence band versus realised accuracy — the project's headline table.

    Bands are formed over the *decision confidence* max(p, 1-p) rather than
    over p itself, because that is the quantity a human interface would show
    when it says "the model is 90% sure".

    Attributes:
        lower: Lower edge of the confidence band.
        upper: Upper edge of the confidence band.
        count: Rows in the band.
        share: Share of all rows in the band.
        mean_confidence: Mean stated confidence in the band.
        accuracy: Share of rows in the band whose argmax label was correct.
        gap: ``mean_confidence - accuracy``; positive means the model claimed
            more certainty than it earned.

    """

    lower: float
    upper: float
    count: int
    share: float
    mean_confidence: float
    accuracy: float
    gap: float


def _validate(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coerce and check a (labels, probabilities) pair."""
    y_true = np.asarray(y_true).ravel().astype(int)
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


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared difference between predicted probability and outcome.

    Args:
        y_true: Binary labels, 1 = bad credit.
        y_prob: Predicted P(bad credit).

    Returns:
        The Brier score; 0 is perfect, 0.25 is a constant 0.5 predictor.

    """
    y_true, y_prob = _validate(y_true, y_prob)
    return float(np.mean((y_prob - y_true) ** 2))


def _bin_edges(y_prob: np.ndarray, n_bins: int, binning: str) -> np.ndarray:
    """Compute bin edges, either equal-width or equal-count.

    Quantile binning is the default because equal-width bins on a credit score
    leave most bins nearly empty and a handful carrying the whole sample, which
    makes the resulting ECE an artefact of the binning rather than of the model.
    """
    if binning == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if binning == "quantile":
        edges = np.quantile(y_prob, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        return np.unique(edges)
    raise ValueError(f"Unknown binning {binning!r}; use 'quantile' or 'uniform'.")


def reliability_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    config: CalibrationConfig | None = None,
) -> tuple[ReliabilityBin, ...]:
    """Bin predictions and contrast mean confidence with observed frequency.

    Empty bins are dropped rather than reported as zero, so that a reliability
    diagram never draws a point at the origin that no observation supports.

    Args:
        y_true: Binary labels, 1 = bad credit.
        y_prob: Predicted P(bad credit).
        config: Bin count and binning scheme.

    Returns:
        The occupied bins, in ascending probability order.

    """
    config = config or CalibrationConfig()
    y_true, y_prob = _validate(y_true, y_prob)

    edges = _bin_edges(y_prob, config.n_bins, config.binning)
    # Right-closed bins, with the first bin closed on the left so that a
    # prediction of exactly 0.0 is not dropped.
    assignment = np.clip(np.searchsorted(edges, y_prob, side="left") - 1, 0, len(edges) - 2)

    bins: list[ReliabilityBin] = []
    for index in range(len(edges) - 1):
        mask = assignment == index
        count = int(mask.sum())
        if count == 0:
            continue
        bins.append(
            ReliabilityBin(
                lower=float(edges[index]),
                upper=float(edges[index + 1]),
                count=count,
                mean_predicted=float(y_prob[mask].mean()),
                observed_frequency=float(y_true[mask].mean()),
            )
        )
    return tuple(bins)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    config: CalibrationConfig | None = None,
) -> float:
    """Count-weighted mean absolute gap between confidence and frequency.

    Args:
        y_true: Binary labels, 1 = bad credit.
        y_prob: Predicted P(bad credit).
        config: Bin count and binning scheme.

    Returns:
        ECE in [0, 1]; lower is better.

    """
    bins = reliability_bins(y_true, y_prob, config)
    total = sum(b.count for b in bins)
    return float(sum(b.count * abs(b.gap) for b in bins) / total)


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    config: CalibrationConfig | None = None,
) -> float:
    """Largest absolute gap between confidence and frequency over occupied bins.

    Exposed as a standalone function, not only as a field of
    :class:`CalibrationMetrics`, so that
    :mod:`credit_confidence_gap.evaluation.null_model` can compute its null
    distribution through the same code path that produces the reported value.

    Args:
        y_true: Binary labels, 1 = bad credit.
        y_prob: Predicted P(bad credit).
        config: Bin count and binning scheme.

    Returns:
        MCE in [0, 1]; lower is better.

    """
    bins = reliability_bins(y_true, y_prob, config)
    return float(max(abs(b.gap) for b in bins))


def calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    config: CalibrationConfig | None = None,
) -> CalibrationMetrics:
    """Compute the full calibration summary.

    Args:
        y_true: Binary labels, 1 = bad credit.
        y_prob: Predicted P(bad credit).
        config: Bin count and binning scheme.

    Returns:
        Brier score, ECE, MCE, signed calibration error and reliability bins.

    """
    config = config or CalibrationConfig()
    y_true, y_prob = _validate(y_true, y_prob)
    bins = reliability_bins(y_true, y_prob, config)
    total = sum(b.count for b in bins)

    return CalibrationMetrics(
        brier_score=brier_score(y_true, y_prob),
        expected_calibration_error=float(sum(b.count * abs(b.gap) for b in bins) / total),
        maximum_calibration_error=float(max(abs(b.gap) for b in bins)),
        signed_calibration_error=float(sum(b.count * b.gap for b in bins) / total),
        mean_predicted=float(y_prob.mean()),
        observed_frequency=float(y_true.mean()),
        n_bins=config.n_bins,
        binning=config.binning,
        bins=bins,
    )


def trust_gap_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    config: CalibrationConfig | None = None,
) -> tuple[TrustGapRow, ...]:
    """Tabulate stated confidence against realised accuracy, by confidence band.

    This is the operationalisation of the project's guiding question. A human
    told "the model is 90% confident" will calibrate their reliance on that
    number; the relevant quantity is therefore not the average error but the
    error *in the band where confidence is highest*, because that is the band
    in which a person is least likely to check.

    Args:
        y_true: Binary labels, 1 = bad credit.
        y_prob: Predicted P(bad credit).
        config: Supplies ``trust_gap_bins``, the confidence band edges.

    Returns:
        One row per occupied band, in ascending confidence order.

    """
    config = config or CalibrationConfig()
    y_true, y_prob = _validate(y_true, y_prob)

    confidence = np.maximum(y_prob, 1.0 - y_prob)
    predicted_label = (y_prob >= 0.5).astype(int)
    correct = (predicted_label == y_true).astype(float)

    edges = np.asarray(config.trust_gap_bins, dtype=float)
    rows: list[TrustGapRow] = []
    for index in range(len(edges) - 1):
        lower, upper = float(edges[index]), float(edges[index + 1])
        is_last = index == len(edges) - 2
        mask = (confidence >= lower) & (confidence <= upper if is_last else confidence < upper)
        count = int(mask.sum())
        if count == 0:
            continue
        mean_confidence = float(confidence[mask].mean())
        accuracy = float(correct[mask].mean())
        rows.append(
            TrustGapRow(
                lower=lower,
                upper=upper,
                count=count,
                share=count / y_true.size,
                mean_confidence=mean_confidence,
                accuracy=accuracy,
                gap=mean_confidence - accuracy,
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class PooledTrustGapRow:
    """One confidence band, pooled across repeated three-way splits.

    The reason this exists: on a single split the top confidence band holds
    roughly fifty cases, so a gap of five percentage points is about two and a
    half cases — one standard error. That is a direction, not an observation,
    and it is far too thin to put in front of a human as "at this confidence
    the model has historically been wrong this often". Pooling the twenty
    stability splits raises the band to several hundred cases.

    The interval is deliberately **not** a binomial interval on the pooled
    count. The splits overlap — the same original applicant appears in several
    test partitions — so those cases are not independent and a binomial
    interval would be too narrow. Instead the interval is a cluster bootstrap
    that resamples whole *splits*, which respects the dependence and also
    absorbs the variation from refitting the model on each split.

    Attributes:
        lower: Lower edge of the confidence band.
        upper: Upper edge of the confidence band.
        count: Total cases in the band across all splits.
        n_sources: Splits that contributed a non-empty band.
        mean_confidence: Count-weighted mean stated confidence.
        accuracy: Count-weighted realised accuracy.
        gap: ``mean_confidence - accuracy``.
        gap_lower: Lower bound of the cluster-bootstrap interval on the gap.
        gap_upper: Upper bound of that interval.
        confidence_level: Nominal coverage of the interval.
        per_source_gaps: The individual split-level gaps, ascending. Reported
            because their spread is the honest picture of how much a single
            split could have misled.

    """

    lower: float
    upper: float
    count: int
    n_sources: int
    mean_confidence: float
    accuracy: float
    gap: float
    gap_lower: float
    gap_upper: float
    confidence_level: float
    per_source_gaps: tuple[float, ...]

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval separates the gap from zero."""
        return self.gap_lower > 0.0 or self.gap_upper < 0.0

    def as_dict(self) -> dict[str, object]:
        """Return a serialisable dictionary including the derived verdict."""
        return asdict(self) | {"excludes_zero": self.excludes_zero}

    def __str__(self) -> str:
        """One-line summary suitable for console output."""
        return (
            f"{self.lower:.1f}-{self.upper:.1f}  n={self.count:>5}  "
            f"stated {self.mean_confidence:.3f}  realised {self.accuracy:.3f}  "
            f"gap {self.gap:+.3f} [{self.gap_lower:+.3f}, {self.gap_upper:+.3f}]"
        )


def weighted_band_gap(rows: Sequence[TrustGapRow]) -> tuple[float, float, float, int]:
    """Combine trust-gap rows for one band into a count-weighted summary.

    Public because :mod:`credit_confidence_gap.evaluation.null_model` must pool its
    simulated draws by exactly the same arithmetic that pooled the observed
    value. Two implementations of this would be free to drift, and the
    comparison between them would quietly stop meaning anything.

    Args:
        rows: Trust-gap rows for a single band, one per split.

    Returns:
        ``(mean_confidence, accuracy, gap, total_count)``, count-weighted.

    """
    total = sum(row.count for row in rows)
    confidence = sum(row.count * row.mean_confidence for row in rows) / total
    accuracy = sum(row.count * row.accuracy for row in rows) / total
    return confidence, accuracy, confidence - accuracy, total


def pool_trust_gap(
    tables: Sequence[Sequence[TrustGapRow]],
    config: BootstrapConfig | None = None,
) -> tuple[PooledTrustGapRow, ...]:
    """Combine trust-gap tables from repeated splits into one table.

    Args:
        tables: One trust-gap table per split.
        config: Resample count, coverage and seed for the cluster bootstrap.

    Returns:
        One pooled row per occupied band, in ascending confidence order.

    Raises:
        ValueError: If no tables are supplied.

    """
    config = config or BootstrapConfig()
    if not tables:
        raise ValueError("Cannot pool an empty collection of trust-gap tables.")

    by_band: dict[tuple[float, float], list[TrustGapRow]] = {}
    for table in tables:
        for row in table:
            by_band.setdefault((row.lower, row.upper), []).append(row)

    rng = np.random.default_rng(config.random_state)
    alpha = (1.0 - config.confidence_level) / 2.0
    pooled: list[PooledTrustGapRow] = []

    for (lower, upper), rows in sorted(by_band.items()):
        confidence, accuracy, gap, total = weighted_band_gap(rows)

        # Cluster bootstrap: the resampling unit is a whole split, not a case.
        indices = np.arange(len(rows))
        draws = np.empty(config.n_resamples, dtype=float)
        for i in range(config.n_resamples):
            picked = [rows[j] for j in rng.choice(indices, size=indices.size, replace=True)]
            draws[i] = weighted_band_gap(picked)[2]

        pooled.append(
            PooledTrustGapRow(
                lower=lower,
                upper=upper,
                count=total,
                n_sources=len(rows),
                mean_confidence=confidence,
                accuracy=accuracy,
                gap=gap,
                gap_lower=float(np.quantile(draws, alpha)),
                gap_upper=float(np.quantile(draws, 1.0 - alpha)),
                confidence_level=config.confidence_level,
                per_source_gaps=tuple(sorted(row.gap for row in rows)),
            )
        )
    return tuple(pooled)
