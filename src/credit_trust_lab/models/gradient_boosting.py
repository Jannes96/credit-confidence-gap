"""Gradient boosting: the stronger learner, and the more confident one.

Included as the contrast case rather than as an upgrade. Boosted trees
typically buy a modest amount of discrimination on tabular data of this size
and pay for it with sharper, less well calibrated probabilities. That
combination — better ranking, worse honesty — is precisely the situation in
which an explanation is most likely to talk a human into misplaced trust, and
is therefore the object of study, not a defect to be tuned away.

Depth and learning rate are held deliberately modest: with 600 training rows,
an unconstrained booster memorises the sample and its confidence becomes
meaningless.
"""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier

from credit_trust_lab.models.base import CreditModel


class GradientBoostingScorer(CreditModel):
    """Gradient boosted decision trees on the shared design matrix."""

    name = "gradient_boosting"

    def _build_estimator(self) -> GradientBoostingClassifier:
        """Return the configured gradient boosting classifier."""
        return GradientBoostingClassifier(
            n_estimators=self.config.gb_n_estimators,
            learning_rate=self.config.gb_learning_rate,
            max_depth=self.config.gb_max_depth,
            subsample=self.config.gb_subsample,
            random_state=self.config.random_state,
        )
