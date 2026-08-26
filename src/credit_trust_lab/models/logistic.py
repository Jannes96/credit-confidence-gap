"""Logistic regression: the credit industry's default, and this project's baseline.

Regulated scorecards are logistic regressions for a reason that matters here:
the contribution of each characteristic is a single additive term, which makes
an adverse-action reason code derivable rather than reconstructed. That the
model is also naturally well calibrated on the log-odds scale makes it the
reference point against which the gradient boosting model's confidence is
judged.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from credit_trust_lab.models.base import CreditModel


class LogisticScorecard(CreditModel):
    """L2-penalised logistic regression on the shared design matrix."""

    name = "logistic_regression"

    def _build_estimator(self) -> LogisticRegression:
        """Return the configured logistic regression.

        ``class_weight`` is intentionally left at ``None``: weighting the loss
        would shift the predicted probabilities away from the sample
        distribution and corrupt every calibration metric reported later.

        The penalty is scikit-learn's default L2, set implicitly: passing
        ``penalty="l2"`` is deprecated from scikit-learn 1.8 onwards.
        """
        return LogisticRegression(
            C=self.config.logistic_c,
            solver="lbfgs",
            max_iter=self.config.logistic_max_iter,
            class_weight=None,
            random_state=self.config.random_state,
        )
