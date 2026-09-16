"""Check whether the explanation describes what the model actually does.

Stability asks whether an explanation is reproducible. This module asks the
harder question: whether the characteristics it names are the ones the model
actually responds to.

The test is a perturbation. Take the features an explanation ranks highest for
a case, replace them with an uninformative baseline value, and see how far the
model's judgement moves. Then do the same with randomly chosen features. If the
explanation is faithful, the features it named should move the prediction
substantially further than arbitrary ones. If they do not, the explanation is
telling a story about a model that is doing something else — the most dangerous
possible input to a human decision, because it is both wrong and persuasive.

Two properties are measured, and they can fail independently:

* **Magnitude.** Do the named features move the prediction more than random
  ones? Reported as a ratio; 1.0 means the explanation carries no information
  about the model's sensitivities.
* **Direction.** Removing a feature credited with *increasing* risk should
  *decrease* the predicted risk. A high magnitude with poor direction
  agreement means the explanation finds the right characteristics and gets
  their effect backwards.

Everything is measured on the log-odds scale, where SHAP attributions live and
where contributions are additive. On the probability scale the sigmoid would
compress movement near 0 and 1, making identical model behaviour look weaker
for confident cases.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from credit_confidence_gap.config import ExplanationConfig
from credit_confidence_gap.data.codebook import FEATURES, FeatureKind
from credit_confidence_gap.explanation.shap_values import Attributions, model_margin
from credit_confidence_gap.models.base import CreditModel


@dataclass(frozen=True)
class FaithfulnessReport:
    """How well an explanation predicts the model's own sensitivities.

    Attributes:
        model_name: Model the explanation belongs to.
        top_k: Number of features masked in the headline comparison.
        top_k_effect: Mean absolute log-odds shift when the top-k attributed
            features are replaced by their baseline values.
        random_k_effect: The same for k randomly chosen features.
        effect_ratio: ``top_k_effect / random_k_effect``. Above 1 means the
            explanation identifies features the model is genuinely sensitive
            to; 1.0 means it identifies nothing.
        direction_agreement: Share of cases where masking the top feature moved
            the prediction in the direction its attribution predicted.
        deletion_curve: Mean absolute shift after masking the 1st, 2nd, … k-th
            most attributed feature cumulatively. A faithful explanation gives
            a curve that rises steeply and then flattens.
        n_cases: Cases evaluated.

    """

    model_name: str
    top_k: int
    top_k_effect: float
    random_k_effect: float
    effect_ratio: float
    direction_agreement: float
    deletion_curve: tuple[float, ...]
    n_cases: int

    def as_dict(self) -> dict[str, object]:
        """Return a serialisable dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        """One-line summary suitable for console output."""
        return (
            f"{self.model_name}: top-{self.top_k} shift {self.top_k_effect:.3f} vs "
            f"random {self.random_k_effect:.3f} (ratio {self.effect_ratio:.2f}), "
            f"direction {self.direction_agreement:.2f}"
        )


def baseline_row(X_reference: pd.DataFrame) -> pd.Series:
    """Return the uninformative value of each feature, from a reference sample.

    Numeric features use the median; every coded feature uses the mode. Using
    the mode rather than, say, the lowest code matters: masking must move a
    case towards the *typical* applicant, not towards an extreme one, or the
    perturbation would measure distance from an unusual corner of the feature
    space instead of the contribution of the feature.

    Args:
        X_reference: Rows to draw baseline values from — the training
            partition, never the explained cases.

    Returns:
        One baseline value per codebook feature.

    """
    values: dict[str, float] = {}
    for feature in FEATURES:
        column = X_reference[feature.name]
        if feature.kind is FeatureKind.NUMERIC:
            values[feature.name] = float(column.median())
        else:
            values[feature.name] = float(column.mode().iloc[0])
    return pd.Series(values)


def _mask(
    X: pd.DataFrame, baseline: pd.Series, feature_index: np.ndarray, names: tuple[str, ...]
) -> pd.DataFrame:
    """Replace the indicated features with baseline values, per row.

    Args:
        X: Cases to perturb.
        baseline: Baseline value per feature.
        feature_index: Array of shape ``(n_cases, k)`` naming, per row, the
            positional feature indices to mask.
        names: Feature names in positional order.

    Returns:
        A copy of ``X`` with the selected cells overwritten.

    """
    masked = X.copy()
    columns = masked.columns.get_indexer(list(names))
    block = masked.to_numpy(copy=True)
    for row, indices in enumerate(feature_index):
        for index in indices:
            block[row, columns[index]] = baseline[names[index]]
    return pd.DataFrame(block, columns=masked.columns, index=masked.index)


def assess_faithfulness(
    model: CreditModel,
    attributions: Attributions,
    X_explain: pd.DataFrame,
    X_reference: pd.DataFrame,
    config: ExplanationConfig | None = None,
) -> FaithfulnessReport:
    """Perturb the explained features and check the model responds accordingly.

    Args:
        model: The fitted model the attributions came from.
        attributions: Attributions to test.
        X_explain: The explained cases, in codebook form.
        X_reference: Rows supplying baseline values, normally the training
            partition.
        config: Top-k and seed.

    Returns:
        The faithfulness report.

    """
    config = config or ExplanationConfig()
    rng = np.random.default_rng(config.random_state)

    names = attributions.feature_names
    baseline = baseline_row(X_reference)
    original = model_margin(model, X_explain)

    ranked = np.argsort(-np.abs(attributions.values), axis=1)
    n_cases, n_features = attributions.values.shape
    k = min(config.top_k, n_features)

    top_masked = _mask(X_explain, baseline, ranked[:, :k], names)
    top_effect = np.abs(model_margin(model, top_masked) - original)

    random_choice = np.stack(
        [rng.choice(n_features, size=k, replace=False) for _ in range(n_cases)]
    )
    random_masked = _mask(X_explain, baseline, random_choice, names)
    random_effect = np.abs(model_margin(model, random_masked) - original)

    # Direction: masking a risk-increasing feature must lower the margin.
    best = ranked[:, 0]
    single_masked = _mask(X_explain, baseline, ranked[:, :1], names)
    single_shift = model_margin(model, single_masked) - original
    claimed = attributions.values[np.arange(n_cases), best]
    moved = np.abs(single_shift) > 1e-9
    direction = (
        float((np.sign(single_shift[moved]) == -np.sign(claimed[moved])).mean())
        if moved.any()
        else float("nan")
    )

    curve = []
    for depth in range(1, k + 1):
        masked = _mask(X_explain, baseline, ranked[:, :depth], names)
        curve.append(float(np.abs(model_margin(model, masked) - original).mean()))

    mean_random = float(random_effect.mean())
    return FaithfulnessReport(
        model_name=attributions.model_name,
        top_k=k,
        top_k_effect=float(top_effect.mean()),
        random_k_effect=mean_random,
        effect_ratio=float(top_effect.mean() / mean_random)
        if mean_random > 1e-12
        else float("inf"),
        direction_agreement=direction,
        deletion_curve=tuple(curve),
        n_cases=n_cases,
    )
