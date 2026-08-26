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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn

from credit_trust_lab import __version__
from credit_trust_lab.config import FIGURES_DIR, METRICS_DIR, Config
from credit_trust_lab.data.download import DATA_SHA256, DATASET_URL, download_raw
from credit_trust_lab.data.loader import data_card, load_dataset
from credit_trust_lab.data.splits import make_splits
from credit_trust_lab.evaluation.calibration import calibration_metrics
from credit_trust_lab.evaluation.discrimination import discrimination_metrics
from credit_trust_lab.evaluation.report import ModelEvaluation, evaluate_model, write_json
from credit_trust_lab.models.registry import build_all_models
from credit_trust_lab.plots import figures


def _print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{text}\n{'-' * len(text)}")


def run_stability(config: Config) -> dict[str, dict[str, list[float]]]:
    """Refit and re-evaluate across many random three-way splits.

    A single split of 1000 rows is one draw from a wide distribution. This run
    reports the spread so that a difference between the two models can be
    judged against the noise of the split itself, rather than presented as a
    result.

    Args:
        config: Full project configuration.

    Returns:
        ``{metric_name: {model_name: [values across seeds]}}``.

    """
    X, y = load_dataset()
    records: dict[str, dict[str, list[float]]] = {
        "ROC-AUC (test)": {},
        "Brier score (test)": {},
        "ECE (test)": {},
    }
    for seed in config.stability.seeds:
        splits = make_splits(X, y, config.split, random_state=seed)
        for model in build_all_models(config.model):
            model.fit(splits.X_train, splits.y_train)
            probabilities = model.predict_proba_bad(splits.X_test)
            truth = np.asarray(splits.y_test)
            discrimination = discrimination_metrics(truth, probabilities)
            calibration = calibration_metrics(truth, probabilities, config.calibration)
            records["ROC-AUC (test)"].setdefault(model.name, []).append(discrimination.roc_auc)
            records["Brier score (test)"].setdefault(model.name, []).append(calibration.brier_score)
            records["ECE (test)"].setdefault(model.name, []).append(
                calibration.expected_calibration_error
            )
    return records


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
    evaluations = [
        evaluate_model(model, splits, config) for model in build_all_models(config.model)
    ]
    print(_summarise(evaluations))
    print(
        f"\nTheoretical threshold under {config.cost.cost_false_negative:.0f}:"
        f"{config.cost.cost_false_positive:.0f} costs: "
        f"{evaluations[0].threshold.theoretical_threshold:.3f}"
    )
    for evaluation in evaluations:
        print(
            f"  {evaluation.model_name:<22} empirical {evaluation.threshold.threshold:.3f}  "
            f"gap {evaluation.threshold.threshold_gap:+.3f}"
        )

    _print_header("4. Figures")
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

    if not args.skip_stability:
        _print_header("5. Stability across seeds")
        stability = run_stability(config)
        for metric, by_model in stability.items():
            for name, values in by_model.items():
                array = np.asarray(values)
                print(
                    f"  {metric:<20} {name:<22} "
                    f"median {np.median(array):.3f}  "
                    f"range [{array.min():.3f}, {array.max():.3f}]"
                )
        written.append(figures.plot_stability(stability, args.figures_dir / "stability.png"))
        write_json(stability, args.metrics_dir / "stability.json")

    _print_header("6. Written artefacts")
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
