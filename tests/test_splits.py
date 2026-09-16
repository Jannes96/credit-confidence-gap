"""Tests for the three-way split.

The central assertion is disjointness. Everything else in the project rests on
the claim that no row used to fit a model or a threshold is also used to report
performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_confidence_gap.config import SplitConfig
from credit_confidence_gap.data.splits import SplitError, make_splits


def test_partitions_are_pairwise_disjoint(synthetic_frame, synthetic_target):
    splits = make_splits(synthetic_frame, synthetic_target)
    train = set(splits.train_index.tolist())
    calibration = set(splits.calibration_index.tolist())
    test = set(splits.test_index.tolist())

    assert train & calibration == set()
    assert train & test == set()
    assert calibration & test == set()


def test_partitions_cover_every_row(synthetic_frame, synthetic_target):
    splits = make_splits(synthetic_frame, synthetic_target)
    covered = (
        set(splits.train_index.tolist())
        | set(splits.calibration_index.tolist())
        | set(splits.test_index.tolist())
    )
    assert covered == set(synthetic_frame.index.tolist())
    assert sum(splits.sizes.values()) == len(synthetic_frame)


def test_no_feature_row_appears_in_two_partitions(synthetic_frame, synthetic_target):
    """Guard against index-based disjointness that hides duplicated content."""
    splits = make_splits(synthetic_frame, synthetic_target)
    frames = [splits.X_train, splits.X_calibration, splits.X_test]
    combined = pd.concat(frames)
    assert len(combined) == len(synthetic_frame)
    assert not combined.index.duplicated().any()


def test_fraction_sizes_match_the_configuration(synthetic_frame, synthetic_target):
    config = SplitConfig(train_fraction=0.6, calibration_fraction=0.2, test_fraction=0.2)
    splits = make_splits(synthetic_frame, synthetic_target, config)
    n = len(synthetic_frame)
    assert splits.sizes["test"] == pytest.approx(0.2 * n, abs=1)
    assert splits.sizes["calibration"] == pytest.approx(0.2 * n, abs=1)
    assert splits.sizes["train"] == pytest.approx(0.6 * n, abs=2)


def test_stratification_preserves_the_class_balance(synthetic_frame, synthetic_target):
    splits = make_splits(synthetic_frame, synthetic_target)
    overall = float(synthetic_target.mean())
    for partition, rate in splits.positive_rates.items():
        assert rate == pytest.approx(overall, abs=0.05), partition


def test_same_seed_gives_the_same_split(synthetic_frame, synthetic_target):
    first = make_splits(synthetic_frame, synthetic_target, random_state=7)
    second = make_splits(synthetic_frame, synthetic_target, random_state=7)
    np.testing.assert_array_equal(first.train_index, second.train_index)
    np.testing.assert_array_equal(first.calibration_index, second.calibration_index)
    np.testing.assert_array_equal(first.test_index, second.test_index)


def test_different_seeds_give_different_splits(synthetic_frame, synthetic_target):
    first = make_splits(synthetic_frame, synthetic_target, random_state=7)
    second = make_splits(synthetic_frame, synthetic_target, random_state=8)
    assert set(first.test_index.tolist()) != set(second.test_index.tolist())


def test_split_config_rejects_fractions_that_do_not_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        SplitConfig(train_fraction=0.7, calibration_fraction=0.2, test_fraction=0.2)


def test_split_config_rejects_degenerate_fractions():
    with pytest.raises(ValueError, match="strictly between"):
        SplitConfig(train_fraction=1.0, calibration_fraction=0.0, test_fraction=0.0)


def test_mismatched_lengths_are_rejected(synthetic_frame, synthetic_target):
    with pytest.raises(SplitError, match="rows"):
        make_splits(synthetic_frame, synthetic_target.iloc[:-5])


def test_targets_follow_their_features(synthetic_frame, synthetic_target):
    """The y of a partition must be the y of exactly those rows."""
    splits = make_splits(synthetic_frame, synthetic_target)
    for X_part, y_part in (
        (splits.X_train, splits.y_train),
        (splits.X_calibration, splits.y_calibration),
        (splits.X_test, splits.y_test),
    ):
        np.testing.assert_array_equal(np.asarray(X_part.index), np.asarray(y_part.index))
        np.testing.assert_array_equal(
            np.asarray(y_part), np.asarray(synthetic_target.loc[X_part.index])
        )


def test_stability_seeds_are_deterministic():
    from credit_confidence_gap.config import StabilityConfig

    config = StabilityConfig(n_repeats=5, base_random_state=100)
    assert config.seeds == (100, 101, 102, 103, 104)


@pytest.mark.integration
def test_real_data_split_is_disjoint_and_stratified(real_dataset):
    X, y = real_dataset
    splits = make_splits(X, y)
    assert splits.sizes == {"train": 600, "calibration": 200, "test": 200}
    for rate in splits.positive_rates.values():
        assert rate == pytest.approx(0.30, abs=0.02)
    assert set(splits.train_index) & set(splits.test_index) == set()
