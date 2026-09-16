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

from credit_confidence_gap.evaluation.calibration import (  # noqa: E402
    CalibrationMetrics,
    PooledTrustGapRow,
    TrustGapRow,
)
from credit_confidence_gap.evaluation.costs import ThresholdResult, cost_curve  # noqa: E402
from credit_confidence_gap.evaluation.discrimination import (  # noqa: E402
    pr_curve_data,
    roc_curve_data,
)
from credit_confidence_gap.evaluation.null_model import NullComparison  # noqa: E402
from credit_confidence_gap.evaluation.report import ModelEvaluation  # noqa: E402
from credit_confidence_gap.explanation.faithfulness import FaithfulnessReport  # noqa: E402
from credit_confidence_gap.explanation.stability import StabilityReport  # noqa: E402

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


def plot_recalibration(
    recalibrated: dict[str, Sequence[ModelEvaluation]],
    path: Path,
    *,
    title: str = "Does recalibration move ECE beyond the measurement floor?",
) -> Path:
    """Compare each calibration map against its own noise floor.

    Every method gets its *own* null interval, drawn as a grey range behind its
    bar. They are not interchangeable: isotonic regression collapses the score
    into a few dozen levels, which changes the bin occupancy and therefore
    lowers its floor considerably. Averaging the intervals into one band — the
    obvious shortcut — would compare each method against a threshold that
    belongs to a different method.

    A bar inside its own range has not been shown to be miscalibrated, so a
    shorter bar there is not an improvement but a different draw.

    Args:
        recalibrated: Evaluations per base model, one per calibration method.
        path: Destination file.
        title: Figure title.

    Returns:
        The path written.

    """
    n_models = len(recalibrated)
    fig, axes = plt.subplots(1, n_models, figsize=(5.8 * n_models, 4.8), squeeze=False)

    for ax, (base_name, variants) in zip(axes[0], recalibrated.items(), strict=True):
        methods = [v.model_name.split("+", 1)[1] for v in variants]
        values = [v.test_partition.calibration.expected_calibration_error for v in variants]
        nulls = [v.test_calibration_null["expected_calibration_error"] for v in variants]
        positions = np.arange(len(methods))

        ax.bar(positions, values, width=0.5, color="#3b6ea5", zorder=3, label="measured ECE")
        for position, null in zip(positions, nulls, strict=True):
            ax.add_patch(
                plt.Rectangle(
                    (position - 0.34, null.null_lower),
                    0.68,
                    null.null_upper - null.null_lower,
                    facecolor="grey",
                    alpha=0.25,
                    zorder=1,
                )
            )
            ax.plot(
                [position - 0.34, position + 0.34],
                [null.null_median, null.null_median],
                color="dimgrey",
                linestyle="--",
                linewidth=1,
                zorder=2,
            )
        for position, value in zip(positions, values, strict=True):
            ax.annotate(
                f"{value:.3f}", (position, value + 0.0015), ha="center", fontsize=8, zorder=4
            )

        # Proxy handles: the per-bar patches cannot carry their own legend entry.
        ax.add_patch(
            plt.Rectangle(
                (0, 0), 0, 0, facecolor="grey", alpha=0.25, label="that method's own 90% null range"
            )
        )
        ax.plot([], [], color="dimgrey", linestyle="--", linewidth=1, label="its null median")

        ax.set_xticks(positions)
        ax.set_xticklabels(methods)
        ax.set_ylim(0, max(max(values), max(n.null_upper for n in nulls)) * 1.35)
        ax.set_ylabel("Expected calibration error (test)")
        ax.set_title(base_name, fontsize=10)
        ax.legend(loc="upper center", fontsize=7, ncol=1)
        ax.grid(alpha=0.3, axis="y", zorder=0)

    fig.suptitle(title, fontsize=11)
    return _save(fig, path)


def plot_explanation_checks(
    stability: dict[str, StabilityReport],
    faithfulness: dict[str, FaithfulnessReport],
    path: Path,
    *,
    title: str = "Can the explanation be believed?",
) -> Path:
    """Show stability against its chance floor, and faithfulness against random masking.

    Two panels because the two properties fail independently. An explanation can
    be faithful to a model whose reasoning is itself an artefact of the sample,
    which is the most misleading combination and the one this project expects to
    find.

    Args:
        stability: Stability reports keyed by model name.
        faithfulness: Faithfulness reports keyed by model name.
        path: Destination file.
        title: Figure title.

    Returns:
        The path written.

    """
    names = list(stability)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    positions = np.arange(len(names))
    width = 0.38

    left = axes[0]
    left.bar(
        positions - width / 2,
        [stability[n].top_k_jaccard for n in names],
        width,
        label="measured top-k overlap",
        color="#3b6ea5",
    )
    left.bar(
        positions + width / 2,
        [stability[n].random_jaccard for n in names],
        width,
        label="chance level",
        color="#b9c6d6",
    )
    for i, name in enumerate(names):
        left.annotate(
            f"top feature same\nin {stability[name].top_feature_agreement:.0%} of cases",
            (i, stability[name].top_k_jaccard + 0.04),
            ha="center",
            fontsize=8,
        )
    left.axhline(1.0, color="green", linestyle="--", linewidth=1, label="fully reproducible")
    left.set_ylim(0, 1.25)
    left.set_xticks(positions)
    left.set_xticklabels(names, fontsize=9)
    left.set_ylabel("Top-k set overlap across bootstrap refits")
    left.set_title("Stability — does the same explanation come back?", fontsize=10)
    left.legend(loc="upper left", fontsize=8)
    left.grid(alpha=0.3, axis="y")

    right = axes[1]
    right.bar(
        positions - width / 2,
        [faithfulness[n].top_k_effect for n in names],
        width,
        label="masking the named features",
        color="#3b6ea5",
    )
    right.bar(
        positions + width / 2,
        [faithfulness[n].random_k_effect for n in names],
        width,
        label="masking random features",
        color="#b9c6d6",
    )
    for i, name in enumerate(names):
        right.annotate(
            f"ratio {faithfulness[name].effect_ratio:.2f}",
            (i, faithfulness[name].top_k_effect + 0.05),
            ha="center",
            fontsize=8,
        )
    right.set_ylim(0, max(faithfulness[n].top_k_effect for n in names) * 1.25)
    right.set_xticks(positions)
    right.set_xticklabels(names, fontsize=9)
    right.set_ylabel("Mean absolute shift in log-odds")
    right.set_title("Faithfulness — does the model respond to them?", fontsize=10)
    right.legend(loc="upper left", fontsize=8)
    right.grid(alpha=0.3, axis="y")

    fig.suptitle(title, fontsize=11)
    return _save(fig, path)


def plot_pooled_trust_gap(
    pooled: dict[str, Sequence[PooledTrustGapRow]],
    nulls: dict[str, dict[tuple[float, float], NullComparison]],
    path: Path,
    *,
    title: str = "Stated confidence versus realised accuracy, pooled across 20 splits",
) -> Path:
    """Draw the gap per confidence band with its interval and its noise floor.

    The project's headline figure. Three things have to be visible at once for
    it to be read honestly: the gap itself, how precisely it is known (the
    cluster-bootstrap interval), and how large a gap this measurement would
    have produced with nothing wrong (the grey null band). A gap whose interval
    overlaps the null band has not been shown.

    Args:
        pooled: Pooled rows per model.
        nulls: Null comparisons per model, keyed by band.
        path: Destination file.
        title: Figure title.

    Returns:
        The path written.

    """
    n_models = len(pooled)
    fig, axes = plt.subplots(1, n_models, figsize=(6.4 * n_models, 5.0), squeeze=False)

    for ax, (name, rows) in zip(axes[0], pooled.items(), strict=True):
        model_nulls = nulls.get(name, {})
        positions = np.arange(len(rows))

        for position, row in zip(positions, rows, strict=True):
            null = model_nulls.get((row.lower, row.upper))
            if null is not None:
                ax.add_patch(
                    plt.Rectangle(
                        (position - 0.32, null.null_lower),
                        0.64,
                        null.null_upper - null.null_lower,
                        facecolor="grey",
                        alpha=0.25,
                        zorder=1,
                    )
                )

        gaps = np.array([row.gap for row in rows])
        lower = gaps - np.array([row.gap_lower for row in rows])
        upper = np.array([row.gap_upper for row in rows]) - gaps
        ax.errorbar(
            positions,
            gaps,
            yerr=np.vstack([lower, upper]),
            fmt="o",
            color="#3b6ea5",
            capsize=5,
            linewidth=1.6,
            markersize=7,
            zorder=3,
            label="pooled gap, 95% cluster bootstrap",
        )
        ax.axhline(0.0, color="black", linewidth=1, zorder=2)
        ax.add_patch(
            plt.Rectangle(
                (0, 0),
                0,
                0,
                facecolor="grey",
                alpha=0.25,
                label="90% range if perfectly calibrated",
            )
        )

        ax.set_xticks(positions)
        ax.set_xticklabels(
            [f"{row.lower:.1f}-{row.upper:.1f}\nn={row.count}" for row in rows], fontsize=8
        )
        ax.set_xlabel("Confidence band  max(p, 1-p)")
        ax.set_ylabel("Stated confidence minus realised accuracy")
        ax.set_title(name, fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3, axis="y", zorder=0)

    fig.suptitle(title, fontsize=11)
    fig.text(
        0.5,
        0.005,
        "Positive means the model claimed more certainty than it earned. "
        "Splits overlap, so the interval is a cluster bootstrap over splits.",
        ha="center",
        fontsize=7,
        style="italic",
    )
    return _save(fig, path)
