"""Fitting a calibrator without letting the threshold see its training data.

The problem this module exists to solve
---------------------------------------
Step 1 established one rule: a fitted quantity must be fitted on an earlier
partition than the one it is reported on. Recalibration threatens to break that
rule in a way that is easy to miss.

The calibrator is fitted on the calibration partition. The decision threshold
is *also* chosen on the calibration partition. If both use the same 200 rows,
the threshold is selected on probabilities that have already absorbed those
rows' outcomes — the calibrated probabilities look better there than they will
anywhere else, and the threshold is tuned to a fit that will not generalise.
The resulting cost on test is then worse than the calibration partition
promised, and nothing in the output says why.

The obvious fix is a fourth partition. A thousand rows cannot pay for one.

The fix used here
-----------------
Cross-fitting. The calibration partition is split into k folds; for each fold,
a calibrator is fitted on the other k-1 folds and applied to the held-out fold.
Every row therefore receives a calibrated probability from a calibrator that
never saw its outcome. Those out-of-fold probabilities are what the threshold
is chosen on, and what the calibration partition's own metrics are computed
from.

A final calibrator is then refitted on all 200 rows and used for the test
partition, because there is no reason to deploy a map fitted on 160 rows when
200 are available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from credit_confidence_gap.config import RecalibrationConfig
from credit_confidence_gap.data.splits import DataSplits
from credit_confidence_gap.models.base import CreditModel
from credit_confidence_gap.recalibration.base import CalibratedModel, Calibrator
from credit_confidence_gap.recalibration.methods import build_calibrator


@dataclass(frozen=True)
class FittedCalibration:
    """A calibrated model plus the honest probabilities used to tune it.

    Attributes:
        calibrated_model: Base model wrapped with the final calibrator, fitted
            on the whole calibration partition. Safe to apply to test.
        calibrator_name: Which map was used.
        out_of_fold_probabilities: Calibrated probabilities for the calibration
            partition, each produced by a calibrator that did not see that
            row's outcome. The threshold is chosen from these.
        raw_probabilities: The base model's uncalibrated probabilities on the
            calibration partition, kept for before/after comparison.
        n_folds: Folds used for cross-fitting.

    """

    calibrated_model: CalibratedModel
    calibrator_name: str
    out_of_fold_probabilities: np.ndarray
    raw_probabilities: np.ndarray
    n_folds: int

    @property
    def n_output_levels(self) -> int:
        """Distinct probability values the map leaves on the calibration partition.

        A step function such as isotonic regression merges applicants into
        ties. Two applicants on the same level can no longer be ranked against
        each other, which is a loss no calibration metric reports. Compare
        against ``len(raw_probabilities)`` to see how much of the score scale
        survived.
        """
        return int(np.unique(self.out_of_fold_probabilities).size)


def cross_fitted_probabilities(
    raw_probabilities: np.ndarray,
    y: np.ndarray,
    calibrator_name: str,
    config: RecalibrationConfig | None = None,
) -> np.ndarray:
    """Produce out-of-fold calibrated probabilities.

    Args:
        raw_probabilities: Uncalibrated P(bad credit) on the calibration
            partition.
        y: Outcomes on that partition, 1 = bad credit.
        calibrator_name: Registry key of the calibration map.
        config: Fold count and seed.

    Returns:
        Calibrated probabilities of the same length, each from a calibrator
        fitted without that row.

    Raises:
        ValueError: If the inputs disagree in length or a fold would contain
            only one class.

    """
    config = config or RecalibrationConfig()
    raw_probabilities = np.asarray(raw_probabilities, dtype=float).ravel()
    y = np.asarray(y).ravel().astype(int)
    if raw_probabilities.shape != y.shape:
        raise ValueError(f"Shape mismatch: probabilities {raw_probabilities.shape} vs y {y.shape}.")
    if len(np.unique(y)) < 2:
        raise ValueError("Cross-fitting needs both classes present.")

    out_of_fold = np.empty_like(raw_probabilities)
    splitter = StratifiedKFold(
        n_splits=config.n_cross_fit_folds, shuffle=True, random_state=config.random_state
    )
    for fit_index, held_out_index in splitter.split(raw_probabilities.reshape(-1, 1), y):
        calibrator = build_calibrator(calibrator_name)
        calibrator.fit(raw_probabilities[fit_index], y[fit_index])
        out_of_fold[held_out_index] = calibrator.transform(raw_probabilities[held_out_index])
    return out_of_fold


def fit_calibration(
    model: CreditModel,
    splits: DataSplits,
    calibrator_name: str,
    config: RecalibrationConfig | None = None,
) -> FittedCalibration:
    """Fit a calibration map for an already-trained model.

    The base model is **not** refitted. It was trained on the training
    partition and stays that way; only the probability map is learned here,
    from the calibration partition.

    Args:
        model: A fitted base model.
        splits: The three partitions.
        calibrator_name: Registry key of the calibration map.
        config: Fold count and seed.

    Returns:
        The wrapped model together with its out-of-fold probabilities.

    Raises:
        RuntimeError: If the base model has not been fitted.

    """
    config = config or RecalibrationConfig()
    if not model.is_fitted:
        raise RuntimeError(
            f"Base model {model.name!r} must be fitted on the training partition "
            "before a calibrator can be fitted on the calibration partition."
        )

    raw = model.predict_proba_bad(splits.X_calibration)
    y_calibration = np.asarray(splits.y_calibration)

    out_of_fold = cross_fitted_probabilities(raw, y_calibration, calibrator_name, config)

    final_calibrator: Calibrator = build_calibrator(calibrator_name)
    final_calibrator.fit(raw, y_calibration)

    return FittedCalibration(
        calibrated_model=CalibratedModel(model, final_calibrator),
        calibrator_name=calibrator_name,
        out_of_fold_probabilities=out_of_fold,
        raw_probabilities=raw,
        n_folds=config.n_cross_fit_folds,
    )


def fit_all_calibrations(
    model: CreditModel,
    splits: DataSplits,
    config: RecalibrationConfig | None = None,
) -> list[FittedCalibration]:
    """Fit every configured calibration map for one model.

    Args:
        model: A fitted base model.
        splits: The three partitions.
        config: Which maps to fit, fold count and seed.

    Returns:
        One result per configured method, in configuration order. The identity
        map comes first by convention, so the control condition is the
        reference row in every table.

    """
    config = config or RecalibrationConfig()
    return [fit_calibration(model, splits, name, config) for name in config.methods]


def calibration_comparison_frame(results: list[FittedCalibration]) -> pd.DataFrame:
    """Summarise how far each map moved the probabilities.

    A map that barely moves anything cannot have fixed anything either; this
    frame makes that visible before any metric is consulted.

    Args:
        results: Fitted calibrations for one model.

    Returns:
        One row per method with the mean and maximum absolute shift.

    """
    rows = []
    for result in results:
        shift = np.abs(result.out_of_fold_probabilities - result.raw_probabilities)
        rows.append(
            {
                "method": result.calibrator_name,
                "mean_abs_shift": float(shift.mean()),
                "max_abs_shift": float(shift.max()),
                "mean_probability_before": float(result.raw_probabilities.mean()),
                "mean_probability_after": float(result.out_of_fold_probabilities.mean()),
                "n_output_levels": result.n_output_levels,
                "n_input_levels": int(np.unique(result.raw_probabilities).size),
            }
        )
    return pd.DataFrame(rows)
