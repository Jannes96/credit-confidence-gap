"""Central, explicit configuration.

Every number that could silently change a result lives here, not in a call site.
Configuration is plain Python dataclasses rather than YAML on purpose: the values
are then type-checked, importable from tests, and impossible to typo into a
silently-ignored key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
METRICS_DIR: Path = REPORTS_DIR / "metrics"

DOCS_DIR: Path = PROJECT_ROOT / "docs"
# The demo page is published, not merely generated: GitHub Pages serves this
# directory straight from the branch, so no build workflow is needed. That is
# also why it sits here rather than under reports/, which stays entirely
# generated-and-ignored.
DEMO_DIR: Path = DOCS_DIR / "demo"


@dataclass(frozen=True)
class SplitConfig:
    """Three-way stratified split: train / calibration / test.

    The calibration set exists so that probability recalibration (step 2) can
    never touch the data used to report performance. It is carved out here, in
    the data layer, so that leakage is prevented by structure rather than by
    discipline.

    With n=1000 and 300 defaults, a 20% test set holds ~60 defaults. Point
    estimates on that basis are noisy, which is why every reported metric is
    accompanied by a bootstrap interval (see ``BootstrapConfig``) and by a
    multi-seed stability run (see ``StabilityConfig``).
    """

    train_fraction: float = 0.60
    calibration_fraction: float = 0.20
    test_fraction: float = 0.20
    random_state: int = 20240601

    def __post_init__(self) -> None:
        """Validate that the three fractions form a partition."""
        total = self.train_fraction + self.calibration_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Split fractions must sum to 1.0, got {total!r}.")
        for name, value in (
            ("train_fraction", self.train_fraction),
            ("calibration_fraction", self.calibration_fraction),
            ("test_fraction", self.test_fraction),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between 0 and 1, got {value!r}.")


@dataclass(frozen=True)
class CostConfig:
    """Cost matrix driving the decision threshold.

    Units are deliberately dimensionless. The default 5:1 ratio is the cost
    matrix shipped with the original Statlog German Credit problem: granting
    credit to a bad risk is held to be five times as expensive as refusing a
    good one. That ratio is an assumption about the lender, not a fact in the
    data, so ``scripts/reproduce_all.py`` additionally sweeps
    ``sensitivity_ratios`` to show how far the resulting decision moves when the
    assumption moves.

    Naming follows the modelling convention used throughout this project:
    the positive class (y=1) is *bad credit*, i.e. the default event.

    Attributes:
        cost_false_negative: Cost of predicting "good" for an actually bad
            credit (a missed default). Statlog default: 5.
        cost_false_positive: Cost of predicting "bad" for an actually good
            credit (a rejected good customer). Statlog default: 1.
        cost_true_negative: Cost of a correctly accepted good credit.
        cost_true_positive: Cost of a correctly refused bad credit.
        sensitivity_ratios: FN/FP ratios explored in the sensitivity analysis.

    """

    cost_false_negative: float = 5.0
    cost_false_positive: float = 1.0
    cost_true_negative: float = 0.0
    cost_true_positive: float = 0.0
    sensitivity_ratios: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0)

    def __post_init__(self) -> None:
        """Reject cost structures for which a threshold is not well defined."""
        if self.cost_false_negative <= self.cost_true_positive:
            raise ValueError(
                "cost_false_negative must exceed cost_true_positive, "
                "otherwise refusing a bad credit is not preferable to granting it."
            )
        if self.cost_false_positive <= self.cost_true_negative:
            raise ValueError(
                "cost_false_positive must exceed cost_true_negative, "
                "otherwise accepting a good credit is not preferable to refusing it."
            )


@dataclass(frozen=True)
class CalibrationConfig:
    """Settings for calibration measurement (not yet for recalibration)."""

    n_bins: int = 10
    binning: str = "quantile"  # "quantile" or "uniform"

    # Bands for the trust-gap table. They start at 0.5 because decision
    # confidence is max(p, 1-p) and can never fall below it.
    trust_gap_bins: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


@dataclass(frozen=True)
class NullModelConfig:
    """Parametric bootstrap under the null hypothesis of perfect calibration.

    Answers the question every calibration number needs answered before it can
    be read: what would this metric have shown if the model were exactly right,
    at this sample size and this score distribution? On 200 test rows the
    answer is "about 0.07 ECE", which is larger than most miscalibration worth
    worrying about — so the floor is reported alongside every value.
    """

    n_resamples: int = 2000
    random_state: int = 20240605


@dataclass(frozen=True)
class RecalibrationConfig:
    """Post-hoc probability recalibration, fitted on the calibration partition.

    ``n_cross_fit_folds`` is the reason this project can claim its
    recalibrated threshold is honest. The calibrator is fitted on the
    calibration partition; choosing the decision threshold on the *same*
    partition would let the threshold see outcomes the calibrator already
    absorbed. Out-of-fold probabilities break that loop without spending a
    fourth data partition, which 1000 rows cannot afford.
    """

    methods: tuple[str, ...] = ("identity", "platt", "isotonic")
    n_cross_fit_folds: int = 5
    random_state: int = 20240606


@dataclass(frozen=True)
class ExplanationConfig:
    """SHAP attribution, plus the two checks that decide whether to believe it.

    Attributes:
        n_background: Rows sampled from the training partition as the SHAP
            background distribution.
        n_explain: Test rows to explain. All 200 is cheap for these models.
        n_stability_refits: Bootstrap refits used to ask whether an
            explanation survives resampling the training data.
        top_k: Number of top-attributed features used in the faithfulness
            check and in the stability rank agreement.
        random_state: Seed for background sampling and bootstrap refits.

    """

    n_background: int = 100
    n_explain: int = 200
    n_stability_refits: int = 25
    top_k: int = 5
    random_state: int = 20240607


@dataclass(frozen=True)
class BootstrapConfig:
    """Percentile bootstrap for confidence intervals on test metrics."""

    n_resamples: int = 2000
    confidence_level: float = 0.95
    random_state: int = 20240602


@dataclass(frozen=True)
class StabilityConfig:
    """Repeated three-way splits, to report spread instead of a single draw."""

    n_repeats: int = 20
    base_random_state: int = 20240603

    @property
    def seeds(self) -> tuple[int, ...]:
        """Deterministic seed sequence used by the stability run."""
        return tuple(self.base_random_state + i for i in range(self.n_repeats))


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for the two baseline models.

    Note the deliberate absence of ``class_weight="balanced"``. Reweighting the
    loss would distort the predicted probabilities, which are the very object of
    study here. Class asymmetry is handled where it belongs: at the decision
    threshold, via ``CostConfig``.
    """

    random_state: int = 20240604

    logistic_c: float = 1.0
    logistic_max_iter: int = 2000

    gb_n_estimators: int = 200
    gb_learning_rate: float = 0.05
    gb_max_depth: int = 3
    gb_subsample: float = 1.0


@dataclass(frozen=True)
class Config:
    """Top-level configuration object passed through the pipeline."""

    split: SplitConfig = field(default_factory=SplitConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    null_model: NullModelConfig = field(default_factory=NullModelConfig)
    recalibration: RecalibrationConfig = field(default_factory=RecalibrationConfig)
    explanation: ExplanationConfig = field(default_factory=ExplanationConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


DEFAULT_CONFIG = Config()
