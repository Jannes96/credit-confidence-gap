"""Tests for the coding table.

The codebook is the project's contract with the raw file. These tests exist
because a wrong level assignment produces a model that trains cleanly, scores
well, and means something other than what its author believes.
"""

from __future__ import annotations

import pandas as pd
import pytest

from credit_confidence_gap.data.codebook import (
    FEATURES,
    FEATURES_BY_NAME,
    RAW_COLUMN_ORDER,
    RENAME_MAP,
    TARGET,
    FeatureKind,
    describe_level,
    feature_names,
)


def test_twenty_predictors_plus_target():
    assert len(FEATURES) == 20
    assert len(RAW_COLUMN_ORDER) == 21
    assert RAW_COLUMN_ORDER[-1] == TARGET.raw_name


def test_names_are_unique():
    assert len({f.name for f in FEATURES}) == len(FEATURES)
    assert len({f.raw_name for f in FEATURES}) == len(FEATURES)


def test_every_categorical_feature_declares_levels():
    for feature in FEATURES:
        if feature.kind is FeatureKind.NUMERIC:
            assert feature.levels == {}
        else:
            assert len(feature.levels) >= 2, feature.name


def test_binary_features_have_exactly_two_levels():
    for feature in FEATURES:
        if feature.kind is FeatureKind.BINARY:
            assert len(feature.levels) == 2, feature.name


def test_kind_partition_covers_all_features():
    total = sum(len(feature_names(kind)) for kind in FeatureKind)
    assert total == len(FEATURES)


def test_foreign_worker_uses_the_corrected_coding():
    """Pin the orientation that distinguishes this edition from the legacy UCI one.

    In the corrected South German Credit data, gastarb 1 = yes and 2 = no. The
    legacy UCI file inverts these labels. If a refactor ever reintroduces the
    legacy orientation, this test is the alarm.
    """
    feature = FEATURES_BY_NAME["foreign_worker"]
    assert feature.raw_name == "gastarb"
    assert feature.levels[1] == "yes"
    assert feature.levels[2] == "no"


def test_people_liable_uses_the_corrected_coding():
    feature = FEATURES_BY_NAME["people_liable"]
    assert feature.levels[1] == "3 or more"
    assert feature.levels[2] == "0 to 2"


def test_purpose_declares_the_unobserved_level():
    """Level 7 (education) exists in the coding table but has zero rows."""
    purpose = FEATURES_BY_NAME["purpose"]
    assert purpose.codes == tuple(range(11))
    assert purpose.levels[7] == "education"
    assert purpose.note is not None


def test_installment_rate_note_records_the_inverted_scale():
    feature = FEATURES_BY_NAME["installment_rate"]
    assert feature.levels[1] == ">= 35"
    assert feature.levels[4] == "< 20"
    assert feature.note is not None


def test_features_with_unknown_levels_carry_a_note():
    """Absence-coded lowest levels must not pass silently as scale minima."""
    for name in ("status", "savings", "property"):
        feature = FEATURES_BY_NAME[name]
        assert feature.note is not None, name
        assert "unknown" in feature.levels[1].lower() or "no " in feature.levels[1].lower()


def test_protected_attributes_are_flagged():
    for name in ("personal_status_sex", "foreign_worker"):
        assert FEATURES_BY_NAME[name].note is not None


def test_rename_map_covers_every_raw_column():
    assert set(RENAME_MAP) == set(RAW_COLUMN_ORDER)


def test_target_encoding_flips_to_default_event():
    """Raw 1 = good must become y = 0; raw 0 = bad must become y = 1."""
    raw = pd.Series([1, 0, 1, 0])
    encoded = TARGET.encode(raw)
    assert list(encoded) == [0, 1, 0, 1]


def test_describe_level_round_trips():
    assert describe_level("savings", 5) == ">= 1000 DM"
    with pytest.raises(KeyError):
        describe_level("savings", 99)
    with pytest.raises(KeyError):
        describe_level("age", 1)  # numeric, no labels


def test_levels_are_immutable():
    with pytest.raises(TypeError):
        FEATURES_BY_NAME["housing"].levels[4] = "castle"  # type: ignore[index]


@pytest.mark.integration
def test_real_data_contains_only_declared_codes(real_dataset):
    """Every code in the actual file must be declared in the codebook."""
    X, _ = real_dataset
    for feature in FEATURES:
        if feature.kind is FeatureKind.NUMERIC:
            continue
        observed = set(X[feature.name].unique().tolist())
        assert observed <= set(feature.codes), (
            f"{feature.name}: undeclared {observed - set(feature.codes)}"
        )


@pytest.mark.integration
def test_real_data_matches_the_documented_sample_design(real_dataset):
    X, y = real_dataset
    assert len(X) == 1000
    assert X.shape[1] == 20
    assert int(y.sum()) == 300  # bad credits, oversampled by design


@pytest.mark.integration
def test_foreign_worker_minority_is_the_foreign_one(real_dataset):
    """A substantive check that the corrected orientation is the loaded one.

    Under the corrected coding, foreign workers are the small minority of this
    South German bank's 1970s loan book. Under the legacy UCI labels the same
    column would claim ~96% foreign workers, which is the tell that the wrong
    edition is in use.
    """
    X, _ = real_dataset
    share_foreign = float((X["foreign_worker"] == 1).mean())
    assert share_foreign < 0.10, (
        f"foreign worker share {share_foreign:.3f} looks like the legacy coding"
    )


@pytest.mark.integration
def test_purpose_level_seven_is_empty_in_the_real_data(real_dataset):
    """The declared-but-unobserved level, pinned so the design matrix stays stable."""
    X, _ = real_dataset
    assert int((X["purpose"] == 7).sum()) == 0
    assert set(X["purpose"].unique()) == set(range(11)) - {7}
