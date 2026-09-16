"""Three-way stratified split: train, calibration, test.

The calibration set is created here, in the data layer, and not inside a model
wrapper. Recalibrating and evaluating on the same rows is the most common
defect in projects of this kind; making the third set a first-class object that
every consumer receives turns that defect from a discipline problem into a
structural impossibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from credit_confidence_gap.config import SplitConfig


class SplitError(RuntimeError):
    """Raised when a split violates its own invariants."""


@dataclass(frozen=True)
class DataSplits:
    """The three disjoint partitions, plus the indices that produced them.

    Attributes:
        X_train: Predictors used to fit models.
        y_train: Targets used to fit models.
        X_calibration: Held out for probability recalibration (step 2). Not to
            be used for reporting.
        y_calibration: Targets of the calibration partition.
        X_test: Touched only for final reporting.
        y_test: Targets of the test partition.
        train_index: Row labels of the training partition.
        calibration_index: Row labels of the calibration partition.
        test_index: Row labels of the test partition.
        random_state: Seed that produced this split.

    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_calibration: pd.DataFrame
    y_calibration: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_index: np.ndarray
    calibration_index: np.ndarray
    test_index: np.ndarray
    random_state: int

    @property
    def sizes(self) -> dict[str, int]:
        """Row count of each partition."""
        return {
            "train": len(self.train_index),
            "calibration": len(self.calibration_index),
            "test": len(self.test_index),
        }

    @property
    def positive_rates(self) -> dict[str, float]:
        """Share of bad credits in each partition."""
        return {
            "train": float(self.y_train.mean()),
            "calibration": float(self.y_calibration.mean()),
            "test": float(self.y_test.mean()),
        }


def _assert_disjoint_and_exhaustive(splits: DataSplits, n_rows: int) -> None:
    """Verify the partition property at runtime, not only in tests."""
    parts = {
        "train": set(splits.train_index.tolist()),
        "calibration": set(splits.calibration_index.tolist()),
        "test": set(splits.test_index.tolist()),
    }
    names = list(parts)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = parts[left] & parts[right]
            if overlap:
                raise SplitError(
                    f"Partitions {left!r} and {right!r} share {len(overlap)} rows: "
                    f"{sorted(overlap)[:10]}..."
                )
    total = sum(len(part) for part in parts.values())
    if total != n_rows:
        raise SplitError(f"Partitions cover {total} rows but the dataset has {n_rows}.")


def make_splits(
    X: pd.DataFrame,
    y: pd.Series,
    config: SplitConfig | None = None,
    *,
    random_state: int | None = None,
) -> DataSplits:
    """Partition the data into train, calibration and test, stratified on ``y``.

    Args:
        X: Predictor frame.
        y: Binary target, 1 = bad credit.
        config: Split fractions and default seed.
        random_state: Overrides ``config.random_state``; used by the multi-seed
            stability run.

    Returns:
        The three partitions and their row indices.

    Raises:
        SplitError: If the resulting partitions overlap or fail to cover the data.

    """
    config = config or SplitConfig()
    seed = config.random_state if random_state is None else random_state

    if len(X) != len(y):
        raise SplitError(f"X has {len(X)} rows but y has {len(y)}.")

    # First cut: hold out the test partition.
    train_calib_idx, test_idx = train_test_split(
        np.asarray(X.index),
        test_size=config.test_fraction,
        stratify=np.asarray(y),
        random_state=seed,
    )

    # Second cut: split the remainder into train and calibration. The fraction
    # is expressed relative to the remainder so that the three shares match the
    # configured fractions of the *original* dataset.
    remaining_fraction = config.train_fraction + config.calibration_fraction
    calibration_share = config.calibration_fraction / remaining_fraction
    train_idx, calibration_idx = train_test_split(
        train_calib_idx,
        test_size=calibration_share,
        stratify=np.asarray(y.loc[train_calib_idx]),
        random_state=seed,
    )

    splits = DataSplits(
        X_train=X.loc[train_idx],
        y_train=y.loc[train_idx],
        X_calibration=X.loc[calibration_idx],
        y_calibration=y.loc[calibration_idx],
        X_test=X.loc[test_idx],
        y_test=y.loc[test_idx],
        train_index=np.asarray(train_idx),
        calibration_index=np.asarray(calibration_idx),
        test_index=np.asarray(test_idx),
        random_state=seed,
    )
    _assert_disjoint_and_exhaustive(splits, n_rows=len(X))
    return splits
