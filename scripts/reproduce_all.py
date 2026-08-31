"""Regenerate every figure and every number in the report.

    python scripts/reproduce_all.py

Downloads and verifies the raw data if needed, fits both models on the training
partition, chooses each model's threshold on the calibration partition,
evaluates on the untouched test partition, runs a multi-seed stability check,
and writes everything to ``reports/``.

Nothing in this project is allowed to depend on a notebook having been run.
"""

from __future__ import annotations

import argparse
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn

from credit_trust_lab import __version__
from credit_trust_lab.config import FIGURES_DIR, METRICS_DIR, Config
from credit_trust_lab.data.download import DATA_SHA256, DATASET_URL, download_raw
from credit_trust_lab.data.loader import data_card, load_dataset
from credit_trust_lab.data.splits import DataSplits, make_splits
from credit_trust_lab.evaluation.calibration import (
    PooledTrustGapRow,
    TrustGapRow,
    calibration_metrics,
    pool_trust_gap,
    trust_gap_table,
)
from credit_trust_lab.evaluation.discrimination import discrimination_metrics
from credit_trust_lab.evaluation.null_model import NullComparison, pooled_trust_gap_null
from credit_trust_lab.evaluation.report import ModelEvaluation, evaluate_model, write_json
from credit_trust_lab.explanation import (
    FaithfulnessReport,
    StabilityReport,
    additivity_error,
    assess_faithfulness,
    assess_stability,
    explain,
)
from credit_trust_lab.models.base import CreditModel
from credit_trust_lab.models.registry import build_all_models
from credit_trust_lab.plots import figures
from credit_trust_lab.recalibration import FittedCalibration, fit_calibration


def _print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{text}\n{'-' * len(text)}")


def run_recalibration(
    models: list[CreditModel], splits: DataSplits, config: Config
) -> dict[str, list[tuple[FittedCalibration, ModelEvaluation]]]:
    """Fit and evaluate every calibration map for every model.

    The base models are already trained; only the probability maps are learned
    here, on the calibration partition. Each variant is then evaluated through
    the *same* ``evaluate_model`` used for raw models, so recalibrated and
    uncalibrated results are directly comparable and no evaluation code had to
    be adapted.

    The out-of-fold probabilities are handed to ``evaluate_model`` explicitly.
    Without that, the threshold would be chosen on calibration-partition
    probabilities produced by a calibrator that had already seen those rows'
    outcomes.

    Args:
        models: Fitted base models.
        splits: The three partitions.
        config: Full project configuration.

    Returns:
        Keyed by base-model name: the fitted calibration and its evaluation,
        one pair per configured method, in configuration order (identity
        first). The fitted object is kept because it carries
        ``n_output_levels``, the resolution the score scale retained — a cost
        of recalibration that no calibration metric reports.

    """
    results: dict[str, list[tuple[FittedCalibration, ModelEvaluation]]] = {}
    for model in models:
        variants: list[tuple[FittedCalibration, ModelEvaluation]] = []
        for method in config.recalibration.methods:
            fitted = fit_calibration(model, splits, method, config.recalibration)
            variants.append(
                (
                    fitted,
                    evaluate_model(
                        fitted.calibrated_model,
                        splits,
                        config,
                        fit=False,
                        calibration_probabilities=fitted.out_of_fold_probabilities,
                    ),
                )
            )
        results[model.name] = variants
    return results


def run_explanations(
    models: list[CreditModel], splits: DataSplits, config: Config
) -> tuple[dict[str, StabilityReport], dict[str, FaithfulnessReport], dict[str, float]]:
    """Explain the test cases and subject each explanation to both checks.

    Args:
        models: Fitted base models.
        splits: The three partitions.
        config: Full project configuration.

    Returns:
        Stability reports, faithfulness reports, and the SHAP additivity error
        per model — the last as a wiring check: Shapley values must reconstruct
        the model margin exactly, so anything above floating-point noise means
        the explainer is attached to something other than the model.

    """
    background = splits.X_train.sample(
        min(config.explanation.n_background, len(splits.X_train)),
        random_state=config.explanation.random_state,
    )
    stability: dict[str, StabilityReport] = {}
    faithfulness: dict[str, FaithfulnessReport] = {}
    additivity: dict[str, float] = {}

    for model in models:
        attributions = explain(model, splits.X_test, background, config.explanation)
        additivity[model.name] = additivity_error(model, splits.X_test, attributions)
        stability[model.name] = assess_stability(
            model.name,
            attributions,
            splits,
            splits.X_test,
            background,
            config.explanation,
            config.model,
        )
        faithfulness[model.name] = assess_faithfulness(
            model, attributions, splits.X_test, splits.X_train, config.explanation
        )
    return stability, faithfulness, additivity


@dataclass(frozen=True)
class StabilitySweep:
    """Everything the repeated-split run collects.

    Attributes:
        metrics: ``{metric_name: {model_name: [values across seeds]}}``.
        trust_gap_tables: One trust-gap table per seed, per model.
        test_probabilities: The test-partition probabilities per seed, per
            model. Kept so the pooled table can be held against a null that
            reproduces the same score distributions.

    """

    metrics: dict[str, dict[str, list[float]]]
    trust_gap_tables: dict[str, list[tuple[TrustGapRow, ...]]]
    test_probabilities: dict[str, list[np.ndarray]]


def run_stability(config: Config) -> StabilitySweep:
    """Refit and re-evaluate across many random three-way splits.

    A single split of 1000 rows is one draw from a wide distribution. This run
    reports the spread so that a difference between the two models can be
    judged against the noise of the split itself, rather than presented as a
    result.

    It also carries the per-seed trust-gap tables, which is what makes the
    project's headline question answerable at all. On one split the top
    confidence band holds around fifty cases; pooled across twenty splits it
    holds several hundred.

    Args:
        config: Full project configuration.

    Returns:
        The collected sweep.

    """
    X, y = load_dataset()
    metrics: dict[str, dict[str, list[float]]] = {
        "ROC-AUC (test)": {},
        "Brier score (test)": {},
        "ECE (test)": {},
    }
    trust_gap_tables: dict[str, list[tuple[TrustGapRow, ...]]] = {}
    test_probabilities: dict[str, list[np.ndarray]] = {}

    for seed in config.stability.seeds:
        splits = make_splits(X, y, config.split, random_state=seed)
        for model in build_all_models(config.model):
            model.fit(splits.X_train, splits.y_train)
            probabilities = model.predict_proba_bad(splits.X_test)
            truth = np.asarray(splits.y_test)
            discrimination = discrimination_metrics(truth, probabilities)
            calibration = calibration_metrics(truth, probabilities, config.calibration)
            metrics["ROC-AUC (test)"].setdefault(model.name, []).append(discrimination.roc_auc)
            metrics["Brier score (test)"].setdefault(model.name, []).append(calibration.brier_score)
            metrics["ECE (test)"].setdefault(model.name, []).append(
                calibration.expected_calibration_error
            )
            trust_gap_tables.setdefault(model.name, []).append(
                trust_gap_table(truth, probabilities, config.calibration)
            )
            test_probabilities.setdefault(model.name, []).append(probabilities)

    return StabilitySweep(
        metrics=metrics,
        trust_gap_tables=trust_gap_tables,
        test_probabilities=test_probabilities,
    )


def pool_and_test_trust_gap(
    sweep: StabilitySweep, config: Config
) -> dict[str, tuple[tuple[PooledTrustGapRow, ...], dict[tuple[float, float], NullComparison]]]:
    """Pool the per-seed trust-gap tables and hold each band against its null.

    Args:
        sweep: Output of :func:`run_stability`.
        config: Full project configuration.

    Returns:
        Per model: the pooled rows and the matching null comparisons.

    """
    results = {}
    for name, tables in sweep.trust_gap_tables.items():
        pooled = pool_trust_gap(tables, config.bootstrap)
        nulls = pooled_trust_gap_null(
            pooled,
            sweep.test_probabilities[name],
            config.calibration,
            config.null_model,
        )
        results[name] = (pooled, nulls)
    return results


def _summarise(evaluations: list[ModelEvaluation]) -> str:
    """Render the console summary table."""
    lines = [
        f"{'model':<22}{'ROC-AUC (95% CI)':<26}{'PR-AUC':<10}"
        f"{'Brier':<9}{'ECE':<8}{'thr':<7}{'cost/case':<10}",
        "-" * 92,
    ]
    for evaluation in evaluations:
        test = evaluation.test_partition
        lines.append(
            f"{evaluation.model_name:<22}"
            f"{str(evaluation.roc_auc_interval):<26}"
            f"{test.discrimination.pr_auc:<10.3f}"
            f"{test.calibration.brier_score:<9.3f}"
            f"{test.calibration.expected_calibration_error:<8.3f}"
            f"{evaluation.threshold.threshold:<7.3f}"
            f"{evaluation.test_cost_per_case:<10.3f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the full reproduction pipeline.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` when omitted.

    Returns:
        Process exit code.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-stability",
        action="store_true",
        help="Skip the multi-seed stability run (the slow part).",
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=FIGURES_DIR, help="Where to write figures."
    )
    parser.add_argument(
        "--metrics-dir", type=Path, default=METRICS_DIR, help="Where to write metrics."
    )
    args = parser.parse_args(argv)

    config = Config()

    _print_header("1. Data")
    raw = download_raw()
    print(f"Source:   {DATASET_URL}")
    print(f"Verified: {raw.data_path.name}  sha256={DATA_SHA256[:16]}...")
    X, y = load_dataset(raw)
    print(f"Loaded:   {len(X)} rows, {X.shape[1]} predictors, {int(y.sum())} bad credits")
    (args.metrics_dir).mkdir(parents=True, exist_ok=True)
    (args.metrics_dir / "data_card.txt").write_text(data_card(), encoding="utf-8")

    _print_header("2. Three-way split")
    splits = make_splits(X, y, config.split)
    print(f"Sizes:          {splits.sizes}")
    print(
        "Bad-credit share: "
        + ", ".join(f"{k}={v:.3f}" for k, v in splits.positive_rates.items())
        + "   (stratified; the 0.30 base rate is a sampling artefact)"
    )

    _print_header("3. Fit and evaluate")
    models = build_all_models(config.model)
    for model in models:
        model.fit(splits.X_train, splits.y_train)
    evaluations = [evaluate_model(model, splits, config, fit=False) for model in models]
    print(_summarise(evaluations))
    print(
        f"\nTheoretical threshold under {config.cost.cost_false_negative:.0f}:"
        f"{config.cost.cost_false_positive:.0f} costs: "
        f"{evaluations[0].threshold.theoretical_threshold:.3f}"
    )
    for evaluation in evaluations:
        gap_null = evaluation.threshold_gap_null
        print(
            f"  {evaluation.model_name:<22} empirical {evaluation.threshold.threshold:.3f}  "
            f"gap {evaluation.threshold.threshold_gap:+.3f}  "
            f"(null 90% [{gap_null.null_lower:+.3f}, {gap_null.null_upper:+.3f}], "
            f"{'outside' if gap_null.is_distinguishable else 'within'} noise)"
        )

    _print_header("4. Is any of that distinguishable from noise?")
    print(
        "Each metric against what a perfectly calibrated model would score on\n"
        f"{len(splits.X_test)} rows with the same score distribution.\n"
    )
    print(f"{'model':<22}{'metric':<30}{'observed':<11}{'null median':<13}{'verdict'}")
    print("-" * 88)
    for evaluation in evaluations:
        for name, comparison in evaluation.test_calibration_null.items():
            verdict = "measurable" if comparison.is_distinguishable else "indistinguishable"
            print(
                f"{evaluation.model_name:<22}{name:<30}"
                f"{comparison.observed:<11.3f}{comparison.null_median:<13.3f}{verdict}"
            )

    _print_header("5. Recalibration (step 2)")
    recalibrated = run_recalibration(models, splits, config)
    print(
        f"{'model':<22}{'method':<11}{'ECE':<8}{'null med':<10}{'Brier':<8}"
        f"{'thr':<7}{'cost/case':<11}{'levels':<8}{'verdict'}"
    )
    print("-" * 100)
    for base_name, variants in recalibrated.items():
        for fitted, variant in variants:
            null = variant.test_calibration_null["expected_calibration_error"]
            verdict = "measurable" if null.is_distinguishable else "indistinguishable"
            method = variant.model_name.split("+", 1)[1]
            print(
                f"{base_name:<22}{method:<11}"
                f"{variant.test_partition.calibration.expected_calibration_error:<8.3f}"
                f"{null.null_median:<10.3f}"
                f"{variant.test_partition.calibration.brier_score:<8.3f}"
                f"{variant.threshold.threshold:<7.3f}"
                f"{variant.test_cost_per_case:<11.3f}"
                f"{fitted.n_output_levels:<8}{verdict}"
            )
    print(
        "\nEvery map is monotone, so no applicant overtakes another. Identity and\n"
        "Platt are strictly monotone and leave ROC-AUC untouched. Isotonic is a\n"
        "step function: it merges distinct scores into ties, which discards\n"
        "ranking resolution and moves ROC-AUC. The 'levels' column is what\n"
        "survives of the score scale."
    )

    _print_header("6. Explanations (step 2)")
    explanation_stability, explanation_faithfulness, additivity = run_explanations(
        models, splits, config
    )
    print("SHAP additivity error (must be at floating-point noise):")
    for name, error in additivity.items():
        print(f"  {name:<22} {error:.2e}")
    print("\nStability under bootstrap refits of the training partition:")
    for report in explanation_stability.values():
        print(f"  {report}")
    print("\nFaithfulness — do the named features move the model more than random ones?")
    for report in explanation_faithfulness.values():
        print(f"  {report}")

    _print_header("7. Figures")
    test_curves = {
        e.model_name: (e.test_partition.y_true, e.test_partition.y_prob) for e in evaluations
    }
    calib_curves = {
        e.model_name: (e.calibration_partition.y_true, e.calibration_partition.y_prob)
        for e in evaluations
    }
    written = [
        figures.plot_reliability(
            {e.model_name: e.test_partition.calibration for e in evaluations},
            args.figures_dir / "reliability.png",
        ),
        figures.plot_roc(test_curves, args.figures_dir / "roc.png"),
        figures.plot_precision_recall(test_curves, args.figures_dir / "precision_recall.png"),
        figures.plot_cost_curve(
            calib_curves,
            {e.model_name: e.threshold for e in evaluations},
            args.figures_dir / "cost_curve.png",
        ),
        figures.plot_cost_ratio_sensitivity(
            {e.model_name: e.sensitivity for e in evaluations},
            args.figures_dir / "cost_ratio_sensitivity.png",
        ),
        figures.plot_trust_gap(
            {e.model_name: e.test_partition.trust_gap for e in evaluations},
            args.figures_dir / "trust_gap.png",
        ),
    ]

    written.append(
        figures.plot_recalibration(
            {base: [v for _, v in variants] for base, variants in recalibrated.items()},
            args.figures_dir / "recalibration.png",
        )
    )
    written.append(
        figures.plot_explanation_checks(
            explanation_stability,
            explanation_faithfulness,
            args.figures_dir / "explanation_checks.png",
        )
    )

    if not args.skip_stability:
        _print_header("8. Stability across seeds")
        sweep = run_stability(config)
        for metric, by_model in sweep.metrics.items():
            for name, values in by_model.items():
                array = np.asarray(values)
                print(
                    f"  {metric:<20} {name:<22} "
                    f"median {np.median(array):.3f}  "
                    f"range [{array.min():.3f}, {array.max():.3f}]"
                )
        written.append(figures.plot_stability(sweep.metrics, args.figures_dir / "stability.png"))
        write_json(sweep.metrics, args.metrics_dir / "stability.json")

        _print_header("8b. Trust gap, pooled across the same splits")
        print(
            "The project's headline question. On one split the top band holds ~50\n"
            "cases; pooled it holds several hundred. The interval is a cluster\n"
            "bootstrap over whole splits, because the splits overlap and a binomial\n"
            "interval on the pooled count would be too narrow.\n"
        )
        pooled_gap = pool_and_test_trust_gap(sweep, config)
        for name, (rows, nulls) in pooled_gap.items():
            print(f"  {name}")
            for row in rows:
                null = nulls.get((row.lower, row.upper))
                verdict = (
                    ("measurable" if null.is_distinguishable else "indistinguishable")
                    if null
                    else "no null"
                )
                print(f"    {row}  {verdict}")
        written.append(
            figures.plot_pooled_trust_gap(
                {name: rows for name, (rows, _) in pooled_gap.items()},
                {name: nulls for name, (_, nulls) in pooled_gap.items()},
                args.figures_dir / "trust_gap_pooled.png",
            )
        )
        write_json(
            {
                name: {
                    "pooled": [row.as_dict() for row in rows],
                    "null": {f"{lo:.1f}-{hi:.1f}": c.as_dict() for (lo, hi), c in nulls.items()},
                }
                for name, (rows, nulls) in pooled_gap.items()
            },
            args.metrics_dir / "trust_gap_pooled.json",
        )

    write_json(
        {
            "recalibration": {
                base: [
                    variant.as_dict()
                    | {
                        "n_output_levels": fitted.n_output_levels,
                        "n_input_levels": int(np.unique(fitted.raw_probabilities).size),
                    }
                    for fitted, variant in variants
                ]
                for base, variants in recalibrated.items()
            },
            "explanation_stability": {
                name: report.as_dict() for name, report in explanation_stability.items()
            },
            "explanation_faithfulness": {
                name: report.as_dict() for name, report in explanation_faithfulness.items()
            },
            "shap_additivity_error": additivity,
        },
        args.metrics_dir / "step2.json",
    )

    _print_header("9. Written artefacts")
    write_json(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "package_version": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "config": {
                "split": config.split.__dict__,
                "cost": config.cost.__dict__,
                "calibration": config.calibration.__dict__,
                "model": config.model.__dict__,
            },
            "split_sizes": splits.sizes,
            "split_positive_rates": splits.positive_rates,
            "models": [evaluation.as_dict() for evaluation in evaluations],
        },
        args.metrics_dir / "results.json",
    )
    for path in [*written, args.metrics_dir / "results.json", args.metrics_dir / "data_card.txt"]:
        print(f"  {path}")

    print(
        "\nReminder: probabilities are calibrated against a stratified sample "
        "(300/700) of accepted 1970s applicants. They are not default rates."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
