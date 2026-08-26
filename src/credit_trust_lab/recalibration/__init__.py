"""Post-hoc probability recalibration, fitted on the reserved calibration partition.

Step 2 of the project. Addresses honesty, not discrimination: every map here is
monotone, so ROC-AUC is unchanged by construction and only the calibration
metrics can move.
"""

from credit_trust_lab.recalibration.base import CalibratedModel, Calibrator
from credit_trust_lab.recalibration.fitting import (
    FittedCalibration,
    cross_fitted_probabilities,
    fit_all_calibrations,
    fit_calibration,
)
from credit_trust_lab.recalibration.methods import (
    CALIBRATOR_REGISTRY,
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    build_calibrator,
)

__all__ = [
    "CALIBRATOR_REGISTRY",
    "CalibratedModel",
    "Calibrator",
    "FittedCalibration",
    "IdentityCalibrator",
    "IsotonicCalibrator",
    "PlattCalibrator",
    "build_calibrator",
    "cross_fitted_probabilities",
    "fit_all_calibrations",
    "fit_calibration",
]
