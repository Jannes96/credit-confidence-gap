"""Tests for pooling the trust-gap table across repeated splits.

Pooling exists to turn the project's headline quantity from a direction into
an observation: one split gives a top confidence band of about fifty cases,
twenty splits give several hundred. These tests pin the arithmetic and, more
importantly, the honesty of the interval — the splits overlap, so the interval
must come from resampling whole splits rather than individual cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from credit_confidence_gap.config import BootstrapConfig, CalibrationConfig, NullModelConfig
from credit_confidence_gap.evaluation.calibration import (
    TrustGapRow,
    pool_trust_gap,
    trust_gap_table,
    weighted_band_gap,
)
from credit_confidence_gap.evaluation.null_model import pooled_trust_gap_null

FAST_BOOTSTRAP = BootstrapConfig(n_resamples=200, random_state=7)
FAST_NULL = NullModelConfig(n_resamples=120, random_state=8)


def _row(lower: float, upper: float, count: int, confidence: float, accuracy: float) -> TrustGapRow:
    """Build a trust-gap row directly, so the arithmetic can be checked by hand."""
    return TrustGapRow(
        lower=lower,
        upper=upper,
        count=count,
        share=1.0,
        mean_confidence=confidence,
        accuracy=accuracy,
        gap=confidence - accuracy,
    )


# --------------------------------------------------------------------------
# The arithmetic
# --------------------------------------------------------------------------


def test_weighted_band_gap_matches_a_hand_computation():
    """Ten cases at 90% confidence and 80% accuracy, thirty at 90% and 100%.

    Weighted accuracy = (10*0.8 + 30*1.0) / 40 = 0.95, so the gap is -0.05.
    """
    rows = [_row(0.9, 1.0, 10, 0.9, 0.8), _row(0.9, 1.0, 30, 0.9, 1.0)]
    confidence, accuracy, gap, total = weighted_band_gap(rows)
    assert total == 40
    assert confidence == pytest.approx(0.9)
    assert accuracy == pytest.approx(0.95)
    assert gap == pytest.approx(-0.05)


def test_pooling_sums_counts_and_weights_by_them():
    tables = [[_row(0.9, 1.0, 10, 0.9, 0.8)], [_row(0.9, 1.0, 90, 0.9, 1.0)]]
    pooled = pool_trust_gap(tables, FAST_BOOTSTRAP)
    assert len(pooled) == 1
    assert pooled[0].count == 100
    assert pooled[0].n_sources == 2
    # The large split dominates, so accuracy sits near 0.98, not near 0.9.
    assert pooled[0].accuracy == pytest.approx(0.98)


def test_pooling_a_single_table_reproduces_it():
    table = [_row(0.8, 0.9, 40, 0.85, 0.80), _row(0.9, 1.0, 60, 0.95, 0.90)]
    pooled = pool_trust_gap([table], FAST_BOOTSTRAP)
    assert [row.count for row in pooled] == [40, 60]
    assert [round(row.gap, 6) for row in pooled] == [0.05, 0.05]


def test_bands_are_returned_in_ascending_order():
    tables = [[_row(0.9, 1.0, 5, 0.95, 0.9), _row(0.5, 0.6, 5, 0.55, 0.5)]]
    pooled = pool_trust_gap(tables, FAST_BOOTSTRAP)
    assert [row.lower for row in pooled] == [0.5, 0.9]


def test_bands_missing_from_some_splits_are_still_pooled():
    """A band absent from one split must not disappear, only count fewer sources."""
    tables = [
        [_row(0.5, 0.6, 5, 0.55, 0.5), _row(0.9, 1.0, 5, 0.95, 0.9)],
        [_row(0.9, 1.0, 7, 0.95, 0.9)],
    ]
    pooled = {(row.lower, row.upper): row for row in pool_trust_gap(tables, FAST_BOOTSTRAP)}
    assert pooled[(0.5, 0.6)].n_sources == 1
    assert pooled[(0.9, 1.0)].n_sources == 2


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="empty collection"):
        pool_trust_gap([])


# --------------------------------------------------------------------------
# The interval
# --------------------------------------------------------------------------


def test_interval_brackets_the_point_estimate(rng):
    tables = [
        trust_gap_table(rng.binomial(1, p), p)
        for p in (rng.uniform(0.02, 0.98, 200) for _ in range(20))
    ]
    for row in pool_trust_gap(tables, FAST_BOOTSTRAP):
        assert row.gap_lower <= row.gap <= row.gap_upper


def test_interval_reflects_disagreement_between_splits():
    """Splits that agree give a tight interval; splits that disagree give a wide one.

    This is the property that makes the cluster bootstrap the right choice: it
    is driven by variation *between* splits, which is where the dependence
    lives, not by the pooled case count.
    """
    agreeing = [[_row(0.9, 1.0, 50, 0.95, 0.90)] for _ in range(20)]
    disagreeing = [[_row(0.9, 1.0, 50, 0.95, 0.70 if i % 2 else 1.00)] for i in range(20)]
    tight = pool_trust_gap(agreeing, FAST_BOOTSTRAP)[0]
    wide = pool_trust_gap(disagreeing, FAST_BOOTSTRAP)[0]
    assert (tight.gap_upper - tight.gap_lower) < (wide.gap_upper - wide.gap_lower)


def test_identical_splits_give_a_degenerate_interval():
    """With every split identical there is nothing to resample, so width is zero."""
    tables = [[_row(0.9, 1.0, 50, 0.95, 0.90)] for _ in range(20)]
    row = pool_trust_gap(tables, FAST_BOOTSTRAP)[0]
    assert row.gap_upper - row.gap_lower == pytest.approx(0.0, abs=1e-12)


def test_per_source_gaps_are_recorded_and_sorted():
    tables = [[_row(0.9, 1.0, 10, 0.95, acc)] for acc in (0.8, 1.0, 0.9)]
    row = pool_trust_gap(tables, FAST_BOOTSTRAP)[0]
    assert len(row.per_source_gaps) == 3
    assert list(row.per_source_gaps) == sorted(row.per_source_gaps)


def test_excludes_zero_matches_the_interval():
    clearly_overconfident = [[_row(0.9, 1.0, 50, 0.95, 0.60)] for _ in range(20)]
    row = pool_trust_gap(clearly_overconfident, FAST_BOOTSTRAP)[0]
    assert row.excludes_zero
    assert "excludes_zero" in row.as_dict()


# --------------------------------------------------------------------------
# The null
# --------------------------------------------------------------------------


def test_perfectly_calibrated_splits_are_not_flagged(rng):
    """Outcomes drawn from the stated probabilities must land inside the null."""
    probabilities = [rng.uniform(0.02, 0.98, 200) for _ in range(20)]
    tables = [trust_gap_table(rng.binomial(1, p), p) for p in probabilities]
    pooled = pool_trust_gap(tables, FAST_BOOTSTRAP)
    nulls = pooled_trust_gap_null(pooled, probabilities, CalibrationConfig(), FAST_NULL)

    flagged = sum(nulls[(row.lower, row.upper)].is_distinguishable for row in pooled)
    assert flagged <= 1, "an honest model should rarely fall outside a 90% band"


def test_an_overconfident_model_is_flagged(rng):
    """States 95% confidence everywhere while being right about 60% of the time."""
    probabilities = [np.full(200, 0.95) for _ in range(20)]
    tables = [trust_gap_table(rng.binomial(1, 0.6, 200), p) for p in probabilities]
    pooled = pool_trust_gap(tables, FAST_BOOTSTRAP)
    nulls = pooled_trust_gap_null(pooled, probabilities, CalibrationConfig(), FAST_NULL)
    assert all(nulls[(row.lower, row.upper)].is_distinguishable for row in pooled)


def test_pooling_narrows_the_null_relative_to_one_split(rng):
    """More cases per band must lower the noise floor, or pooling bought nothing."""
    probabilities = [rng.uniform(0.02, 0.98, 200) for _ in range(20)]
    tables = [trust_gap_table(rng.binomial(1, p), p) for p in probabilities]

    one = pool_trust_gap(tables[:1], FAST_BOOTSTRAP)
    many = pool_trust_gap(tables, FAST_BOOTSTRAP)
    null_one = pooled_trust_gap_null(one, probabilities[:1], CalibrationConfig(), FAST_NULL)
    null_many = pooled_trust_gap_null(many, probabilities, CalibrationConfig(), FAST_NULL)

    band = (0.9, 1.0)
    width_one = null_one[band].null_upper - null_one[band].null_lower
    width_many = null_many[band].null_upper - null_many[band].null_lower
    assert width_many < width_one


def test_null_requires_at_least_one_split():
    with pytest.raises(ValueError, match="empty collection"):
        pooled_trust_gap_null([], [], CalibrationConfig(), FAST_NULL)
