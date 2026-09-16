"""Shared fixtures.

Unit tests run on synthetic data so that the suite passes on a clean clone with
no network. Tests that genuinely need the real file are marked ``integration``
and skip themselves when it is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_confidence_gap.config import RAW_DIR
from credit_confidence_gap.data.codebook import FEATURES, FeatureKind
from credit_confidence_gap.data.download import DATA_MEMBER


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator, so failures are reproducible.

    Function-scoped on purpose. A session-scoped generator is consumed in
    whatever order the tests happen to run, so a test can pass alone and fail
    in a full run — the draws it sees depend on which tests ran first. A fresh
    generator per test makes every test reproducible in isolation and immune to
    ordering.
    """
    return np.random.default_rng(1234)


@pytest.fixture
def synthetic_frame(rng: np.random.Generator) -> pd.DataFrame:
    """Return a 400-row frame respecting the codebook's declared level sets."""
    n = 400
    columns: dict[str, np.ndarray] = {}
    for feature in FEATURES:
        if feature.kind is FeatureKind.NUMERIC:
            columns[feature.name] = rng.integers(12, 60, size=n)
        else:
            columns[feature.name] = rng.choice(np.asarray(feature.codes), size=n)
    return pd.DataFrame(columns)


@pytest.fixture
def synthetic_target(synthetic_frame: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Return a target with genuine signal, so AUC-based tests are not degenerate."""
    logit = (
        -1.0
        - 0.9 * (synthetic_frame["status"].to_numpy() - 2.5)
        - 0.5 * (synthetic_frame["savings"].to_numpy() - 3.0)
        + 0.03 * (synthetic_frame["duration"].to_numpy() - 30.0)
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    return pd.Series(rng.binomial(1, probability), name="credit_risk")


@pytest.fixture
def calibrated_predictions(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return a large, perfectly calibrated sample: outcomes drawn from the stated p."""
    probabilities = rng.uniform(0.02, 0.98, size=40_000)
    outcomes = rng.binomial(1, probabilities)
    return outcomes, probabilities


@pytest.fixture(scope="session")
def real_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Return the actual dataset, skipping the test when it has not been downloaded."""
    if not (RAW_DIR / DATA_MEMBER).exists():
        pytest.skip("Raw data absent; run `make data` to enable integration tests.")
    from credit_confidence_gap.data.loader import load_dataset

    return load_dataset()
