"""Feature preparation, driven entirely by the codebook.

Design decisions worth stating explicitly:

* **Ordinal columns stay ordinal.** The integer codes are the P2 scores of
  Häußler (1979), constructed so that larger is better; discarding that order
  would throw away information the data demonstrably carry. They are scaled,
  not one-hot encoded.
* **Categories are declared, not learned.** The one-hot encoder receives the
  level list from the codebook rather than inferring it from the training
  partition. A rare level absent from a 600-row training split therefore still
  gets a column, and no partition can ever produce an "unseen category". The
  price is one all-zero column for the declared-but-unobserved ``purpose``
  level 8, which an L2 penalty drives to zero.
* **No resampling and no class weights.** Class asymmetry is a property of the
  decision, not of the likelihood, and is handled at the threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_trust_lab.data.codebook import FEATURES, FEATURES_BY_NAME, FeatureKind, feature_names


def _one_hot_encoder(columns: list[str]) -> OneHotEncoder:
    """Build a one-hot encoder with categories taken from the codebook."""
    categories = [np.asarray(FEATURES_BY_NAME[name].codes) for name in columns]
    return OneHotEncoder(
        categories=categories,
        drop="first",
        handle_unknown="error",
        sparse_output=False,
        dtype=np.float64,
    )


def build_preprocessor() -> ColumnTransformer:
    """Assemble the column transformer used by every model.

    Both models share this transformer so that any performance difference
    between them is attributable to the estimator and not to the preparation.

    Returns:
        An unfitted :class:`~sklearn.compose.ColumnTransformer`.

    """
    scaled = feature_names(FeatureKind.NUMERIC) + feature_names(FeatureKind.ORDINAL)
    encoded = feature_names(FeatureKind.NOMINAL) + feature_names(FeatureKind.BINARY)

    return ColumnTransformer(
        transformers=[
            ("scaled", Pipeline([("scale", StandardScaler())]), scaled),
            ("encoded", _one_hot_encoder(encoded), encoded),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def expected_feature_names() -> list[str]:
    """Names of the design-matrix columns, in transformer output order.

    Derived from the codebook without fitting anything, so tests can assert the
    design matrix independently of scikit-learn's naming behaviour.

    Returns:
        Column names of the transformed design matrix.

    """
    names = feature_names(FeatureKind.NUMERIC) + feature_names(FeatureKind.ORDINAL)
    for name in feature_names(FeatureKind.NOMINAL) + feature_names(FeatureKind.BINARY):
        feature = FEATURES_BY_NAME[name]
        for code in feature.codes[1:]:  # drop="first"
            names.append(f"{name}={code}")
    return names


def describe_design_matrix(transformed: np.ndarray) -> pd.DataFrame:
    """Wrap a transformed array back into a named frame.

    Args:
        transformed: Output of a fitted preprocessor.

    Returns:
        The same values under the names from :func:`expected_feature_names`.

    Raises:
        ValueError: If the array width does not match the codebook.

    """
    names = expected_feature_names()
    if transformed.shape[1] != len(names):
        raise ValueError(
            f"Design matrix has {transformed.shape[1]} columns "
            f"but the codebook implies {len(names)}."
        )
    return pd.DataFrame(transformed, columns=names)


def validate_frame(X: pd.DataFrame) -> None:
    """Check that a predictor frame matches the codebook before fitting.

    Args:
        X: Predictor frame.

    Raises:
        ValueError: On missing columns, unexpected columns, or missing values.

    """
    expected = {f.name for f in FEATURES}
    actual = set(X.columns)
    if missing := sorted(expected - actual):
        raise ValueError(f"Predictor frame is missing columns: {missing}")
    if unexpected := sorted(actual - expected):
        raise ValueError(f"Predictor frame has undeclared columns: {unexpected}")
    if X.isna().any().any():
        bad = sorted(X.columns[X.isna().any()])
        raise ValueError(f"Predictor frame contains missing values in: {bad}")
