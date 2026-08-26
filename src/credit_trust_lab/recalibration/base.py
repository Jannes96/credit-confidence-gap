"""The calibrator interface, and the wrapper that makes it invisible downstream.

A calibrator is a monotone map from stated probability to corrected
probability, fitted on the calibration partition. Monotonicity means no
applicant can overtake another: the order is never inverted.

That is weaker than "ROC-AUC is unchanged", and the difference matters.
Identity and Platt scaling are *strictly* monotone, so they preserve ROC-AUC
exactly. Isotonic regression is only weakly monotone — it is a step function,
so it maps distinct scores onto identical values. Those ties destroy ranking
resolution between the applicants they merge, and ROC-AUC moves as a result.
On this data isotonic regression collapses 200 distinct calibration scores into
roughly thirty levels, which is a real cost that an ECE comparison alone would
never reveal.

The wrapper is what lets the entire step 1 apparatus be reused unchanged. A
:class:`CalibratedModel` exposes the same ``name`` / ``predict_proba_bad`` /
``decide`` surface as a raw model, so every metric, figure and threshold
routine treats it identically to an uncalibrated one. No evaluation code was
modified to accommodate recalibration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import pandas as pd

from credit_trust_lab.models.base import CreditModel

# Probabilities are clipped away from 0 and 1 before any logit transform.
# Gradient boosting readily produces 1e-6 and 1 - 1e-6; without this, Platt
# scaling would be fitted on infinities.
_EPS = 1e-6


class NotFittedError(RuntimeError):
    """Raised when a calibrator is used before being fitted."""


def _validate_probabilities(p: np.ndarray) -> np.ndarray:
    """Coerce to a 1-D float array in [0, 1] and reject malformed input."""
    p = np.asarray(p, dtype=float).ravel()
    if p.size == 0:
        raise ValueError("Cannot calibrate on an empty sample.")
    if np.isnan(p).any():
        raise ValueError("Probabilities contain NaN.")
    if (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must lie in [0, 1].")
    return p


def logit(p: np.ndarray) -> np.ndarray:
    """Log-odds transform, clipped away from the asymptotes.

    Args:
        p: Probabilities in [0, 1].

    Returns:
        Log-odds of the clipped probabilities.

    """
    clipped = np.clip(_validate_probabilities(p), _EPS, 1.0 - _EPS)
    return np.log(clipped / (1.0 - clipped))


class Calibrator(ABC):
    """Base class for post-hoc probability recalibration.

    Attributes:
        name: Short identifier used in filenames and report tables.

    """

    name: str = "base"

    def __init__(self) -> None:
        """Initialise an unfitted calibrator."""
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self._fitted

    @abstractmethod
    def _fit(self, p: np.ndarray, y: np.ndarray) -> None:
        """Fit the mapping. Subclass hook."""

    @abstractmethod
    def _transform(self, p: np.ndarray) -> np.ndarray:
        """Apply the fitted mapping. Subclass hook."""

    def fit(self, p: np.ndarray, y: np.ndarray) -> Self:
        """Fit the calibration map on held-out probabilities and outcomes.

        Args:
            p: Uncalibrated P(bad credit) on the calibration partition.
            y: Observed outcomes on that same partition, 1 = bad credit.

        Returns:
            The fitted calibrator, for chaining.

        Raises:
            ValueError: If the inputs are malformed or of different lengths.

        """
        p = _validate_probabilities(p)
        y = np.asarray(y).ravel().astype(int)
        if p.shape != y.shape:
            raise ValueError(f"Shape mismatch: p {p.shape} vs y {y.shape}.")
        if not np.isin(y, (0, 1)).all():
            raise ValueError("y must contain only 0 and 1.")
        self._fit(p, y)
        self._fitted = True
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        """Map stated probabilities to corrected ones.

        Args:
            p: Uncalibrated P(bad credit).

        Returns:
            Corrected probabilities, in [0, 1].

        Raises:
            NotFittedError: If the calibrator has not been fitted.

        """
        if not self._fitted:
            raise NotFittedError(f"Calibrator {self.name!r} has not been fitted.")
        corrected = self._transform(_validate_probabilities(p))
        return np.clip(np.asarray(corrected, dtype=float), 0.0, 1.0)

    def __repr__(self) -> str:
        """Show the calibrator name and whether it is fitted."""
        return (
            f"<{type(self).__name__} name={self.name!r} {'fitted' if self._fitted else 'unfitted'}>"
        )


class CalibratedModel:
    """A fitted model plus a fitted calibrator, behaving like a model.

    Duck-types :class:`~credit_trust_lab.models.base.CreditModel` for the two
    operations the evaluation layer uses. It deliberately does **not** subclass
    it: a ``CreditModel`` owns a preprocessing pipeline and an estimator and
    knows how to fit itself from raw features, none of which applies here.
    Inheriting would have meant overriding ``fit`` into something that lies
    about what it does.

    Attributes:
        model: The already-fitted base model. Never refitted here.
        calibrator: The already-fitted calibration map.
        name: Composite name, e.g. ``"gradient_boosting+isotonic"``.

    """

    def __init__(self, model: CreditModel, calibrator: Calibrator) -> None:
        """Pair a fitted model with a fitted calibrator.

        Args:
            model: A fitted :class:`CreditModel`.
            calibrator: A fitted :class:`Calibrator`.

        Raises:
            NotFittedError: If either component is unfitted, which would
                otherwise surface much later as a confusing prediction error.

        """
        if not model.is_fitted:
            raise NotFittedError(f"Base model {model.name!r} must be fitted first.")
        if not calibrator.is_fitted:
            raise NotFittedError(f"Calibrator {calibrator.name!r} must be fitted first.")
        self.model = model
        self.calibrator = calibrator
        self.name = f"{model.name}+{calibrator.name}"

    @property
    def is_fitted(self) -> bool:
        """Always true: both components are fitted at construction."""
        return True

    def predict_proba_bad(self, X: pd.DataFrame) -> np.ndarray:
        """Predict recalibrated P(bad credit).

        Args:
            X: Predictor frame in codebook form.

        Returns:
            Calibrated probabilities of shape ``(n_samples,)``.

        """
        return self.calibrator.transform(self.model.predict_proba_bad(X))

    def decide(self, X: pd.DataFrame, threshold: float) -> np.ndarray:
        """Turn recalibrated probabilities into refuse/accept decisions.

        Args:
            X: Predictor frame in codebook form.
            threshold: Probability at or above which the credit is refused.

        Returns:
            Integer array, 1 = refuse (predicted bad), 0 = accept.

        """
        return (self.predict_proba_bad(X) >= threshold).astype(int)

    def __repr__(self) -> str:
        """Show the composite name."""
        return f"<CalibratedModel name={self.name!r}>"
