"""The three calibration maps, including the one that does nothing.

``identity`` is not a placeholder. It is the control condition. Without it,
"isotonic regression improved ECE from 0.064 to 0.051" has no denominator: the
reader cannot tell whether that is an effect or the ordinary wobble of a metric
measured on 200 rows. Every recalibration result in this project is reported
against the identity map and against the null distribution from
:mod:`credit_trust_lab.evaluation.null_model`.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from credit_trust_lab.recalibration.base import Calibrator, logit


class IdentityCalibrator(Calibrator):
    """Leaves probabilities untouched — the control condition."""

    name = "identity"

    def _fit(self, p: np.ndarray, y: np.ndarray) -> None:
        """Fit nothing; the map is the identity by definition."""

    def _transform(self, p: np.ndarray) -> np.ndarray:
        """Return the input unchanged."""
        return p


class PlattCalibrator(Calibrator):
    """Platt scaling: a one-dimensional logistic regression on the log-odds.

    Fits ``P(bad) = sigmoid(a * logit(p) + b)``. Two parameters, so it is
    stable on 200 rows — the main reason to prefer it here over isotonic
    regression. It can only apply a smooth monotone stretch, so it corrects
    systematic over- or under-confidence but cannot fix a locally wrong region
    of the score scale.

    Fitting on the log-odds rather than on ``p`` directly is what makes the
    identity map reachable: ``a = 1, b = 0`` reproduces the input exactly, so a
    well-calibrated model is left almost alone instead of being distorted.
    """

    name = "platt"

    def __init__(self) -> None:
        """Initialise an unfitted Platt calibrator."""
        super().__init__()
        self._model: LogisticRegression | None = None

    def _fit(self, p: np.ndarray, y: np.ndarray) -> None:
        """Fit the two-parameter logistic map on the log-odds."""
        # C is large so the fit is effectively unpenalised: with two parameters
        # there is nothing to regularise, and shrinking `a` towards zero would
        # flatten every probability towards the base rate.
        self._model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        self._model.fit(logit(p).reshape(-1, 1), y)

    def _transform(self, p: np.ndarray) -> np.ndarray:
        """Apply the fitted sigmoid."""
        assert self._model is not None  # guarded by Calibrator.transform
        return self._model.predict_proba(logit(p).reshape(-1, 1))[:, 1]

    @property
    def coefficients(self) -> tuple[float, float]:
        """The fitted ``(a, b)``.

        Worth inspecting: ``a < 1`` means the model's probabilities were too
        extreme and are being pulled towards the middle, ``a > 1`` that they
        were too timid. ``a ≈ 1, b ≈ 0`` means Platt scaling found nothing to
        correct.

        Returns:
            Slope and intercept on the log-odds scale.

        Raises:
            AssertionError: If accessed before fitting.

        """
        assert self._model is not None, "Calibrator has not been fitted."
        return float(self._model.coef_[0][0]), float(self._model.intercept_[0])


class IsotonicCalibrator(Calibrator):
    """Isotonic regression: a free monotone step function.

    Strictly more flexible than Platt scaling, and correspondingly more prone
    to overfitting. On a 200-row calibration partition it can carve out steps
    supported by a handful of observations, which is exactly the failure mode
    this project is built to notice — hence ``out_of_bounds="clip"`` and the
    identity control.
    """

    name = "isotonic"

    def __init__(self) -> None:
        """Initialise an unfitted isotonic calibrator."""
        super().__init__()
        self._model: IsotonicRegression | None = None

    def _fit(self, p: np.ndarray, y: np.ndarray) -> None:
        """Fit the monotone step function."""
        self._model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self._model.fit(p, y)

    def _transform(self, p: np.ndarray) -> np.ndarray:
        """Apply the fitted step function."""
        assert self._model is not None  # guarded by Calibrator.transform
        return np.asarray(self._model.predict(p), dtype=float)


CALIBRATOR_REGISTRY: dict[str, type[Calibrator]] = {
    IdentityCalibrator.name: IdentityCalibrator,
    PlattCalibrator.name: PlattCalibrator,
    IsotonicCalibrator.name: IsotonicCalibrator,
}


def build_calibrator(name: str) -> Calibrator:
    """Instantiate a registered calibrator by name.

    Args:
        name: Registry key, e.g. ``"isotonic"``.

    Returns:
        An unfitted calibrator.

    Raises:
        KeyError: If ``name`` is not registered.

    """
    if name not in CALIBRATOR_REGISTRY:
        raise KeyError(f"Unknown calibrator {name!r}. Available: {sorted(CALIBRATOR_REGISTRY)}")
    return CALIBRATOR_REGISTRY[name]()
