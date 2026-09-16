"""Check whether the explanation survives resampling the training data.

An explanation is presented to a human as an account of *why* the model
decided. That framing only holds if the account is a property of the problem
rather than of the particular 600 rows the model happened to see. If refitting
on a bootstrap resample reshuffles which characteristics are named, then the
reason code is an artefact of sampling noise — and showing it to a caseworker
manufactures exactly the unwarranted confidence this project studies.

Nothing here asks whether the explanation is *correct*; that is the separate
question handled in :mod:`credit_confidence_gap.explanation.faithfulness`. This
module asks only whether it is *reproducible*, which is the weaker and more
basic requirement. An explanation that fails here cannot pass there in any
meaningful sense.

Only the training partition is resampled. The explained cases and the
background distribution stay fixed, so any variation observed is attributable
to the model refit and not to a change of question.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from credit_confidence_gap.config import ExplanationConfig, ModelConfig
from credit_confidence_gap.data.splits import DataSplits
from credit_confidence_gap.explanation.shap_values import Attributions, explain
from credit_confidence_gap.models.base import CreditModel
from credit_confidence_gap.models.registry import build_model

# Attributions below this magnitude (log-odds) are treated as "no claim made".
# Their sign is arbitrary, so counting sign flips among them would report noise
# as instability and make every explanation look unstable.
_NEGLIGIBLE = 0.01


@dataclass(frozen=True)
class StabilityReport:
    """How much an explanation moves when the training data are resampled.

    Attributes:
        model_name: Model the explanation belongs to.
        n_refits: Bootstrap refits performed.
        top_k: Size of the top-k set compared per case.
        top_k_jaccard: Mean Jaccard overlap between the reference top-k set and
            each refit's, averaged over cases and refits. 1.0 means every refit
            names the same characteristics; 0.5 means half of a reason code
            would change if the bank had sampled a different 600 customers.
        random_jaccard: Expected overlap between two *independently drawn*
            top-k sets. The floor against which ``top_k_jaccard`` must be read:
            with 5 of 20 features it is 0.14, so 0.60 is well above chance and
            still far from reproducible.
        random_top_feature_agreement: Chance-level value of
            ``top_feature_agreement``, namely ``1 / n_features``.
        sign_agreement: Share of non-negligible (case, feature) attributions
            whose sign matches the reference.
        top_feature_agreement: Share of cases where the single most influential
            feature is the same as in the reference fit. The strictest measure,
            and the one closest to what a one-line reason code shows.
        global_rank_correlation: Mean Spearman correlation between the
            reference global importance ranking and each refit's.
        relative_spread: Median across features of (SD of attribution across
            refits) / (mean absolute attribution). A value of 0.5 means the
            typical attribution wobbles by half its own size.
        per_feature_spread: Per-feature standard deviation of the mean
            attribution across refits, descending.

    """

    model_name: str
    n_refits: int
    top_k: int
    top_k_jaccard: float
    random_jaccard: float
    sign_agreement: float
    top_feature_agreement: float
    random_top_feature_agreement: float
    global_rank_correlation: float
    relative_spread: float
    per_feature_spread: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        """Return a serialisable dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        """One-line summary suitable for console output."""
        return (
            f"{self.model_name}: top-{self.top_k} overlap {self.top_k_jaccard:.2f} "
            f"(chance {self.random_jaccard:.2f}), "
            f"same top feature {self.top_feature_agreement:.2f} "
            f"(chance {self.random_top_feature_agreement:.2f}), "
            f"sign agreement {self.sign_agreement:.2f}, "
            f"global rank corr {self.global_rank_correlation:.2f}"
        )


def expected_random_jaccard(k: int, n_features: int) -> float:
    """Return the expected Jaccard overlap of two independently drawn k-subsets.

    Two random k-subsets of n features share ``k^2 / n`` elements in
    expectation, so their expected union is ``2k - k^2 / n``. Reported
    alongside the measured overlap so that no stability figure in this project
    is quoted without the chance level it must beat.

    Args:
        k: Subset size.
        n_features: Number of features to choose from.

    Returns:
        Chance-level Jaccard overlap.

    """
    intersection = k * k / n_features
    return float(intersection / (2 * k - intersection))


def _bootstrap_refit(
    model_name: str,
    splits: DataSplits,
    model_config: ModelConfig,
    rng: np.random.Generator,
) -> CreditModel:
    """Refit the model on a bootstrap resample of the training partition."""
    n = len(splits.X_train)
    picked = rng.integers(0, n, size=n)
    X = splits.X_train.iloc[picked]
    y = splits.y_train.iloc[picked]
    model = build_model(model_name, model_config)
    model.fit(X, y)
    return model


def _top_k_sets(values: np.ndarray, k: int) -> list[set[int]]:
    """Return indices of the k largest absolute attributions, per case."""
    order = np.argsort(-np.abs(values), axis=1)[:, :k]
    return [set(row.tolist()) for row in order]


def assess_stability(
    model_name: str,
    reference: Attributions,
    splits: DataSplits,
    X_explain: pd.DataFrame,
    X_background: pd.DataFrame,
    config: ExplanationConfig | None = None,
    model_config: ModelConfig | None = None,
) -> StabilityReport:
    """Refit on bootstrap resamples and measure how far the explanation moves.

    Args:
        model_name: Registry key of the model to refit.
        reference: Attributions from the model fitted on the full training
            partition — the explanation that would actually be shown.
        splits: The three partitions; only the training one is resampled.
        X_explain: The same cases the reference explained.
        X_background: The same background distribution the reference used.
        config: Refit count, top-k and seed.
        model_config: Hyperparameters, held fixed across refits so that only
            the data vary.

    Returns:
        The stability report.

    """
    config = config or ExplanationConfig()
    model_config = model_config or ModelConfig()
    rng = np.random.default_rng(config.random_state)

    reference_top = _top_k_sets(reference.values, config.top_k)
    reference_best = np.argmax(np.abs(reference.values), axis=1)
    reference_global = np.abs(reference.values).mean(axis=0)

    jaccards: list[float] = []
    sign_matches: list[float] = []
    best_matches: list[float] = []
    rank_correlations: list[float] = []
    refit_values: list[np.ndarray] = []

    for _ in range(config.n_stability_refits):
        refit = _bootstrap_refit(model_name, splits, model_config, rng)
        attributions = explain(refit, X_explain, X_background, config)
        refit_values.append(attributions.values)

        refit_top = _top_k_sets(attributions.values, config.top_k)
        jaccards.extend(
            len(a & b) / len(a | b) for a, b in zip(reference_top, refit_top, strict=True)
        )

        # Sign agreement, restricted to attributions that actually claim something.
        meaningful = (np.abs(reference.values) > _NEGLIGIBLE) & (
            np.abs(attributions.values) > _NEGLIGIBLE
        )
        if meaningful.any():
            same_sign = np.sign(reference.values[meaningful]) == np.sign(
                attributions.values[meaningful]
            )
            sign_matches.append(float(same_sign.mean()))

        best_matches.append(
            float((np.argmax(np.abs(attributions.values), axis=1) == reference_best).mean())
        )
        correlation = spearmanr(
            reference_global, np.abs(attributions.values).mean(axis=0)
        ).statistic
        rank_correlations.append(float(correlation))

    stacked = np.stack(refit_values)  # (refits, cases, features)
    mean_per_refit = np.abs(stacked).mean(axis=1)  # (refits, features)
    spread = mean_per_refit.std(axis=0)
    magnitude = mean_per_refit.mean(axis=0)
    relative = np.divide(spread, magnitude, out=np.zeros_like(spread), where=magnitude > 1e-12)

    return StabilityReport(
        model_name=reference.model_name,
        n_refits=config.n_stability_refits,
        top_k=config.top_k,
        top_k_jaccard=float(np.mean(jaccards)),
        random_jaccard=expected_random_jaccard(config.top_k, len(reference.feature_names)),
        sign_agreement=float(np.mean(sign_matches)) if sign_matches else float("nan"),
        top_feature_agreement=float(np.mean(best_matches)),
        random_top_feature_agreement=1.0 / len(reference.feature_names),
        global_rank_correlation=float(np.mean(rank_correlations)),
        relative_spread=float(np.median(relative)),
        per_feature_spread=dict(
            sorted(
                zip(reference.feature_names, spread.tolist(), strict=True),
                key=lambda item: -item[1],
            )
        ),
    )
