"""Matplotlib figures for the report.

Every figure takes already-computed metric objects rather than a model, so that
plotting can never become a second, divergent implementation of a metric. No
styling beyond Matplotlib defaults, and no seaborn: the figures are evidence,
not decoration.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Figures are written to disk, never shown interactively.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from credit_trust_lab.evaluation.calibration import CalibrationMetrics, TrustGapRow  # noqa: E402
from credit_trust_lab.evaluation.costs import ThresholdResult, cost_curve  # noqa: E402
from credit_trust_lab.evaluation.discrimination import pr_curve_data, roc_curve_data  # noqa: E402

FIGURE_SIZE = (7.0, 5.0)
DPI = 150


def _save(fig: plt.Figure, path: Path) -> Path:
    """Write a figure to disk and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_reliability(
    metrics_by_model: dict[str, CalibrationMetrics],
    path: Path,
    *,
    title: str = "Reliability diagram (test partition)",
) -> Path:
    """Draw observed frequency against mean predicted probability.

    Marker area is proportional to bin occupancy, because a bin holding three
    observations and a bin holding eighty should not read as equally strong
    evidence — the usual reliability diagram hides exactly that.

    Args:
        metrics_by_model: Calibration metrics keyed by model name.
        path: Destination file.
        title: Figure title.

    Returns:
        The path written.

    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="perfect calibration")

    for name, metrics in metrics_by_model.items():
        predicted = [b.mean_predicted for b in metrics.bins]
        observed = [b.observed_frequency for b in metrics.bins]
        counts = np.array([b.count for b in metrics.bins], dtype=float)
        ax.plot(
            predicted,
            observed,
            marker="",
            linewidth=1.2,
            label=f"{name} (ECE {metrics.expected_calibration_error:.3f})",
        )
        ax.scatter(predicted, observed, s=12 + 4 * counts, alpha=0.6)

    ax.set_xlabel("Mean predicted P(bad credit)")
    ax.set_ylabel("Observed frequency of bad credit")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.text(
        0.5,
        0.005,
        "Calibrated against the stratified sample (30% bad by design), not a lending population.",
        ha="center",
        fontsize=7,
        style="italic",
    )
    return _save(fig, path)


def plot_roc(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    path: Path,
    *,
    title: str = "ROC curves (test partition)",
) -> Path:
    """Draw ROC curves for several models.

    Args:
        curves: ``{model_name: (y_true, y_prob)}``.
        path: Destination file.
        title: Figure title.

    Returns:
        The path written.

    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="random")
    for name, (y_true, y_prob) in curves.items():
        data = roc_curve_data(y_true, y_prob)
        ax.plot(data["fpr"], data["tpr"], linewidth=1.4, label=name)
    ax.set_xlabel("False positive rate (good credits refused)")
    ax.set_ylabel("True positive rate (bad credits caught)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_precision_recall(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    path: Path,
    *,
    title: str = "Precision-recall curves (test partition)",
) -> Path:
    """Draw precision-recall curves with the prevalence baseline.

    Args:
        curves: ``{model_name: (y_true, y_prob)}``.
        path: Destination file.
        title: Figure title.

    Returns:
        The path written.

    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    prevalence: float | None = None
    for name, (y_true, y_prob) in curves.items():
        data = pr_curve_data(y_true, y_prob)
        ax.plot(data["recall"], data["precision"], linewidth=1.4, label=name)
        prevalence = float(np.mean(y_true))
    if prevalence is not None:
        ax.axhline(
            prevalence,
            linestyle="--",
            color="grey",
            linewidth=1,
            label=f"prevalence ({prevalence:.2f}, sampled by design)",
        )
    ax.set_xlabel("Recall (share of bad credits caught)")
    ax.set_ylabel("Precision (share of refusals that were bad)")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_cost_curve(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    thresholds: dict[str, ThresholdResult],
    path: Path,
    *,
    title: str = "Cost as a function of the decision threshold (calibration partition)",
) -> Path:
    """Draw realised cost against threshold, marking 0.5 and the theoretical point.

    Args:
        curves: ``{model_name: (y_true, y_prob)}`` on the calibration partition.
        thresholds: Threshold results keyed by model name.
        path: Destination file.
        title: Figure title.

    Returns:
        The path written.

    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for name, (y_true, y_prob) in curves.items():
        grid, costs = cost_curve(y_true, y_prob)
        ax.step(grid, costs / len(y_true), where="post", linewidth=1.3, label=name)
        chosen = thresholds[name].threshold
        ax.axvline(chosen, linestyle=":", linewidth=1, alpha=0.7)

    any_result = next(iter(thresholds.values()))
    ax.axvline(
        any_result.theoretical_threshold,
        color="green",
        linestyle="--",
        linewidth=1,
        label=f"theoretical t* = {any_result.theoretical_threshold:.3f}",
    )
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="naive 0.5")
    ax.set_xlabel("Decision threshold on P(bad credit)")
    ax.set_ylabel("Cost per case (Statlog units)")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_cost_ratio_sensitivity(
    sensitivity_by_model: dict[str, Sequence[ThresholdResult]],
    path: Path,
    *,
    title: str = "Optimal threshold as a function of the assumed cost ratio",
) -> Path:
    """Show how far the recommended decision moves with the cost assumption.

    Args:
        sensitivity_by_model: Threshold sweeps keyed by model name.
        path: Destination file.
        title: Figure title.

    Returns:
        The path written.

    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for name, results in sensitivity_by_model.items():
        ratios = [r.cost_ratio for r in results]
        ax.plot(ratios, [r.threshold for r in results], marker="o", linewidth=1.3, label=name)
    any_results = next(iter(sensitivity_by_model.values()))
    ax.plot(
        [r.cost_ratio for r in any_results],
        [r.theoretical_threshold for r in any_results],
        linestyle="--",
        color="green",
        linewidth=1,
        label="theoretical t* (perfect calibration)",
    )
    ax.axvline(5.0, color="grey", linestyle=":", linewidth=1, label="Statlog default 5:1")
    ax.set_xlabel("Assumed cost ratio C(missed default) / C(rejected good customer)")
    ax.set_ylabel("Cost-minimising threshold")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_trust_gap(
    rows_by_model: dict[str, Sequence[TrustGapRow]],
    path: Path,
    *,
    title: str = "Stated confidence versus realised accuracy (test partition)",
) -> Path:
    """Draw the project's headline comparison, band by band.

    Args:
        rows_by_model: Trust-gap rows keyed by model name.
        path: Destination file.
        title: Figure title.

    Returns:
        The path written.

    """
    n_models = len(rows_by_model)
    fig, axes = plt.subplots(1, n_models, figsize=(6.0 * n_models, 4.6), squeeze=False)

    for ax, (name, rows) in zip(axes[0], rows_by_model.items(), strict=True):
        labels = [f"{r.lower:.1f}-{r.upper:.1f}\n(n={r.count})" for r in rows]
        positions = np.arange(len(rows))
        width = 0.38
        ax.bar(
            positions - width / 2,
            [r.mean_confidence for r in rows],
            width,
            label="stated confidence",
        )
        ax.bar(positions + width / 2, [r.accuracy for r in rows], width, label="realised accuracy")
        for i, row in enumerate(rows):
            ax.annotate(
                f"{row.gap:+.2f}",
                (i, max(row.mean_confidence, row.accuracy) + 0.02),
                ha="center",
                fontsize=8,
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_xlabel("Confidence band  max(p, 1-p)")
        ax.set_ylabel("Share")
        ax.set_title(name, fontsize=10)
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle(title, fontsize=11)
    return _save(fig, path)


def plot_stability(
    records: dict[str, dict[str, list[float]]],
    path: Path,
    *,
    title: str = "Metric spread across 20 random three-way splits",
) -> Path:
    """Box-plot metric distributions across repeated splits.

    Args:
        records: ``{metric_name: {model_name: [values...]}}``.
        path: Destination file.
        title: Figure title.

    Returns:
        The path written.

    """
    metrics = list(records)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.4), squeeze=False)
    for ax, metric in zip(axes[0], metrics, strict=True):
        by_model = records[metric]
        # Set tick labels explicitly rather than via boxplot's keyword, which
        # was renamed from `labels` to `tick_labels` in Matplotlib 3.9.
        ax.boxplot(list(by_model.values()), widths=0.5)
        ax.set_xticks(np.arange(1, len(by_model) + 1))
        ax.set_xticklabels(list(by_model))
        ax.set_title(metric, fontsize=10)
        ax.tick_params(axis="x", labelrotation=15, labelsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(title, fontsize=11)
    return _save(fig, path)
