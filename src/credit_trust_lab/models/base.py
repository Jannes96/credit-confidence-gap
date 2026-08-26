"""The common model interface.

Both baselines are reduced to one contract: given a predictor frame, return
P(bad credit). Everything downstream — evaluation, calibration measurement,
threshold optimisation, and later the explanation and demo layers — depends on
this contract alone, so a new model can be added without touching them, and no
model can be evaluated on terms of its own.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, Self, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from credit_trust_lab.config import ModelConfig
from credit_trust_lab.data.preprocess import build_preprocessor, validate_frame


class NotFittedError(RuntimeError):
    """Raised when predictions are requested from an unfitted model."""


@runtime_checkable
class ScoringModel(Protocol):
    """Anything the evaluation layer is willing to score.

    Narrower than :class:`CreditModel` on purpose: it omits ``fit``, because
    the evaluation layer has no business training anything. A recalibrated
    model (step 2) satisfies this protocol without being a ``CreditModel`` — it
    owns no preprocessing pipeline and cannot be fitted from raw features — and
    that is precisely why the protocol exists rather than a shared base class.

    Attributes:
        name: Short identifier used in filenames and report tables.

    """

    name: str

    @property
    def is_fitted(self) -> bool:
        """Whether the object is ready to produce predictions."""
        ...

    def predict_proba_bad(self, X: pd.DataFrame) -> np.ndarray:
        """Predict P(bad credit) for each row."""
        ...

    def decide(self, X: pd.DataFrame, threshold: float) -> np.ndarray:
        """Turn probabilities into refuse/accept decisions at ``threshold``."""
        ...


class CreditModel(ABC):
    """Base class for every scoring model in this project.

    Subclasses supply an estimator; this class owns the shared preprocessing,
    the fitted-state bookkeeping, and the probability orientation.

    Attributes:
        name: Short identifier used in filenames and report tables.
        config: Hyperparameters and the seed.

    """

    name: str = "base"

    def __init__(self, config: ModelConfig | None = None) -> None:
        """Initialise the model with its configuration.

        Args:
            config: Hyperparameters; defaults are taken from
                :class:`~credit_trust_lab.config.ModelConfig`.

        """
        self.config = config or ModelConfig()
        self._pipeline: Pipeline | None = None

    @abstractmethod
    def _build_estimator(self) -> Any:
        """Return the unfitted scikit-learn estimator for this model."""

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self._pipeline is not None

    @property
    def pipeline(self) -> Pipeline:
        """The fitted pipeline.

        Raises:
            NotFittedError: If the model has not been fitted.

        """
        if self._pipeline is None:
            raise NotFittedError(f"{self.name!r} has not been fitted.")
        return self._pipeline

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> Self:
        """Fit preprocessing and estimator on the training partition.

        Args:
            X: Predictor frame in codebook form.
            y: Binary target, 1 = bad credit.

        Returns:
            The fitted model, for chaining.

        """
        validate_frame(X)
        self._pipeline = Pipeline(
            [("preprocess", build_preprocessor()), ("estimator", self._build_estimator())]
        )
        self._pipeline.fit(X, np.asarray(y))
        return self

    def predict_proba_bad(self, X: pd.DataFrame) -> np.ndarray:
        """Predict P(bad credit) for each row.

        The single probability accessor in the project. Returning one array
        rather than scikit-learn's two-column matrix removes the recurring
        question of which column is which — and with it a whole class of
        silently inverted results.

        Args:
            X: Predictor frame in codebook form.

        Returns:
            Array of shape ``(n_samples,)`` with values in [0, 1].

        """
        validate_frame(X)
        estimator = self.pipeline.named_steps["estimator"]
        proba = self.pipeline.predict_proba(X)
        positive_column = int(np.where(estimator.classes_ == 1)[0][0])
        return np.asarray(proba[:, positive_column], dtype=float)

    def decide(self, X: pd.DataFrame, threshold: float) -> np.ndarray:
        """Turn probabilities into refuse/accept decisions.

        There is deliberately no ``predict`` method: a decision without an
        explicit threshold would silently reinstate 0.5, which this project
        rejects.

        Args:
            X: Predictor frame in codebook form.
            threshold: Probability at or above which the credit is refused.

        Returns:
            Integer array, 1 = refuse (predicted bad), 0 = accept.

        """
        return (self.predict_proba_bad(X) >= threshold).astype(int)

    def __repr__(self) -> str:
        """Show the model name and whether it is fitted."""
        state = "fitted" if self.is_fitted else "unfitted"
        return f"<{type(self).__name__} name={self.name!r} {state}>"
